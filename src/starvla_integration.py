"""
AttnRes integration helpers for real StarVLA backbones.

This module does not replace the underlying Qwen-VL forward pass. Instead, it
adapts the layerwise hidden states produced by the frozen/trainable VLM into a
block-level AttnRes representation that StarVLA action heads can consume.

The resulting representation supports:
1. Full-depth AttnRes routing over VLM depth
2. Uniform skip ablations over block contributions
3. Modality-aware skip ablations over vision/language/action token groups
4. Routing statistics suitable for later VLA analysis
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Optional

import torch
import torch.nn as nn

from .attn_residual import BlockAttnRes


def _get_cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return default if current is None else current


@dataclass
class StarVLAAttnResConfig:
    enabled: bool = False
    n_blocks: int = 8
    temperature: float = 1.0
    enable_skipping: bool = False
    skip_mode: str = "none"  # none | uniform | modality_aware
    uniform_skip_threshold: float = 0.01
    vision_skip_threshold: float = 0.02
    language_skip_threshold: float = 0.01
    action_skip_threshold: float = 0.005


class StarVLAAttnResAdapter(nn.Module):
    """
    Adapt StarVLA/Qwen hidden states into a routed blockwise representation.

    The adapter works over block end-point hidden states extracted from the VLM
    backbone. It is intentionally non-invasive to StarVLA's training loop and
    action heads: action heads still consume a single final hidden tensor.
    """

    MODALITIES = ("vision", "language", "action")

    def __init__(
        self,
        d_model: int,
        num_hidden_layers: int,
        config: StarVLAAttnResConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.n_blocks = min(config.n_blocks, num_hidden_layers)
        self.block_endpoints = self._compute_block_endpoints(num_hidden_layers, self.n_blocks)
        self.block_attn_res = nn.ModuleList(
            [
                BlockAttnRes(
                    d_model=d_model,
                    num_blocks=self.n_blocks,
                    block_idx=block_idx,
                    temperature=config.temperature,
                )
                for block_idx in range(self.n_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

        self.enable_skipping = config.enable_skipping
        self.skip_mode = config.skip_mode
        self.uniform_skip_threshold = config.uniform_skip_threshold
        self.vision_skip_threshold = config.vision_skip_threshold
        self.language_skip_threshold = config.language_skip_threshold
        self.action_skip_threshold = config.action_skip_threshold

    @staticmethod
    def _compute_block_endpoints(num_layers: int, n_blocks: int) -> list[int]:
        base = num_layers // n_blocks
        remainder = num_layers % n_blocks
        endpoints = []
        layer_idx = 0
        for block_idx in range(n_blocks):
            layer_idx += base + (1 if block_idx < remainder else 0)
            endpoints.append(layer_idx)
        return endpoints

    def get_block_layer_ranges(self) -> list[tuple[int, int]]:
        ranges = []
        start = 0
        for endpoint in self.block_endpoints:
            ranges.append((start, endpoint - 1))
            start = endpoint
        return ranges

    def set_inference_config(
        self,
        *,
        enable_skipping: Optional[bool] = None,
        skip_mode: Optional[str] = None,
        uniform_skip_threshold: Optional[float] = None,
        vision_skip_threshold: Optional[float] = None,
        language_skip_threshold: Optional[float] = None,
        action_skip_threshold: Optional[float] = None,
    ) -> None:
        if enable_skipping is not None:
            self.enable_skipping = enable_skipping
        if skip_mode is not None:
            self.skip_mode = skip_mode
        if uniform_skip_threshold is not None:
            self.uniform_skip_threshold = uniform_skip_threshold
        if vision_skip_threshold is not None:
            self.vision_skip_threshold = vision_skip_threshold
        if language_skip_threshold is not None:
            self.language_skip_threshold = language_skip_threshold
        if action_skip_threshold is not None:
            self.action_skip_threshold = action_skip_threshold

    def _select_block_outputs(self, hidden_states: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        outputs = [hidden_states[0]]
        for endpoint in self.block_endpoints:
            outputs.append(hidden_states[endpoint])
        return outputs

    def _build_modality_masks(
        self,
        input_ids: Optional[torch.Tensor],
        image_token_ids: tuple[int, ...] = (),
        action_token_id: Optional[int] = None,
        action_token_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids is None:
            return {}

        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if image_token_ids:
            for token_id in image_token_ids:
                vision_mask |= input_ids == token_id

        action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if action_token_id is not None:
            action_mask |= input_ids == action_token_id
        if action_token_range is not None:
            lo, hi = action_token_range
            action_mask |= (input_ids >= lo) & (input_ids <= hi)

        language_mask = ~(vision_mask | action_mask)
        return {
            "vision": vision_mask,
            "language": language_mask,
            "action": action_mask,
        }

    def _run_full_depth(
        self,
        raw_block_outputs: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        adapted_outputs = [raw_block_outputs[0]]
        routing_weights = []

        for block_idx in range(self.n_blocks):
            routed_input, weights = self.block_attn_res[block_idx](adapted_outputs, return_weights=True)
            adapted_output = self.output_norm(raw_block_outputs[block_idx + 1] + routed_input)
            adapted_outputs.append(adapted_output)
            routing_weights.append(weights)

        return adapted_outputs, routing_weights

    def _compute_importance_stats(
        self,
        routing_weights: list[torch.Tensor],
        modality_masks: dict[str, torch.Tensor],
    ) -> tuple[list[float], dict[str, list[float]]]:
        block_importance = []
        modality_importance = {modality: [] for modality in self.MODALITIES}

        for source_block_idx in range(self.n_blocks):
            source_index = source_block_idx + 1  # index 0 is the embedding source
            downstream = []
            downstream_modality = {modality: [] for modality in self.MODALITIES}

            for target_pos, weights in enumerate(routing_weights):
                if target_pos <= source_block_idx:
                    continue
                if weights.shape[-1] <= source_index:
                    continue

                downstream.append(weights[:, :, source_index].mean().item())
                for modality, mask in modality_masks.items():
                    if mask.any():
                        downstream_modality[modality].append(weights[mask][:, source_index].mean().item())

            block_importance.append(max(downstream) if downstream else 1.0)
            for modality in self.MODALITIES:
                scores = downstream_modality[modality]
                modality_importance[modality].append(max(scores) if scores else 1.0)

        return block_importance, modality_importance

    def _build_keep_mask(
        self,
        block_importance: list[float],
        modality_importance: dict[str, list[float]],
        modality_masks: dict[str, torch.Tensor],
        skip_mode: str,
    ) -> torch.Tensor:
        keep_mask = torch.ones(self.n_blocks, dtype=torch.bool)

        for block_idx in range(self.n_blocks):
            if block_idx == 0 or block_idx == self.n_blocks - 1:
                continue

            if skip_mode == "uniform":
                keep_mask[block_idx] = block_importance[block_idx] >= self.uniform_skip_threshold
                continue

            if skip_mode != "modality_aware":
                continue

            decisions = []
            if modality_masks.get("vision") is not None and modality_masks["vision"].any():
                decisions.append(modality_importance["vision"][block_idx] < self.vision_skip_threshold)
            if modality_masks.get("language") is not None and modality_masks["language"].any():
                decisions.append(modality_importance["language"][block_idx] < self.language_skip_threshold)
            if modality_masks.get("action") is not None and modality_masks["action"].any():
                decisions.append(modality_importance["action"][block_idx] < self.action_skip_threshold)

            keep_mask[block_idx] = not all(decisions) if decisions else True

        return keep_mask

    def compute_online_skip_scores(
        self,
        hidden_states: torch.Tensor,
        routed_input: torch.Tensor,
        modality_masks: dict[str, torch.Tensor],
    ) -> tuple[float, dict[str, float]]:
        delta = (hidden_states - routed_input).abs().mean(dim=-1)
        block_score = float(delta.mean().item())
        modality_scores: dict[str, float] = {}
        for modality in self.MODALITIES:
            mask = modality_masks.get(modality)
            if mask is not None and mask.any():
                modality_scores[modality] = float(delta[mask].mean().item())
            else:
                modality_scores[modality] = block_score
        return block_score, modality_scores

    def should_keep_block_online(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        routed_input: torch.Tensor,
        modality_masks: dict[str, torch.Tensor],
        skip_mode: str,
    ) -> tuple[bool, float, dict[str, float]]:
        block_score, modality_scores = self.compute_online_skip_scores(hidden_states, routed_input, modality_masks)
        if block_idx == 0 or block_idx == self.n_blocks - 1 or skip_mode == "none":
            return True, block_score, modality_scores

        if skip_mode == "uniform":
            return block_score >= self.uniform_skip_threshold, block_score, modality_scores

        decisions = []
        if modality_masks.get("vision") is not None and modality_masks["vision"].any():
            decisions.append(modality_scores["vision"] < self.vision_skip_threshold)
        if modality_masks.get("language") is not None and modality_masks["language"].any():
            decisions.append(modality_scores["language"] < self.language_skip_threshold)
        if modality_masks.get("action") is not None and modality_masks["action"].any():
            decisions.append(modality_scores["action"] < self.action_skip_threshold)
        keep = not all(decisions) if decisions else True
        return keep, block_score, modality_scores

    def _rerun_with_keep_mask(
        self,
        raw_block_outputs: list[torch.Tensor],
        keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[torch.Tensor]]:
        adapted_outputs = [raw_block_outputs[0]]
        executed = []
        executed_weights = []

        for block_idx in range(self.n_blocks):
            if not bool(keep_mask[block_idx]):
                executed.append((block_idx, -1))
                continue

            routed_input, weights = self.block_attn_res[block_idx](adapted_outputs, return_weights=True)
            adapted_output = self.output_norm(raw_block_outputs[block_idx + 1] + routed_input)
            adapted_outputs.append(adapted_output)
            executed.append((block_idx, 0))
            executed_weights.append(weights)

        return adapted_outputs[-1], executed, executed_weights

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        *,
        input_ids: Optional[torch.Tensor] = None,
        image_token_ids: tuple[int, ...] = (),
        action_token_id: Optional[int] = None,
        action_token_range: Optional[tuple[int, int]] = None,
        enable_skipping: Optional[bool] = None,
        skip_mode: Optional[str] = None,
        uniform_skip_threshold: Optional[float] = None,
        vision_skip_threshold: Optional[float] = None,
        language_skip_threshold: Optional[float] = None,
        action_skip_threshold: Optional[float] = None,
        return_routing_info: bool = False,
    ) -> dict[str, Any]:
        if len(hidden_states) < 2:
            raise ValueError("Need embedding state plus transformer hidden states for StarVLA AttnRes")

        if uniform_skip_threshold is not None:
            self.uniform_skip_threshold = uniform_skip_threshold
        if vision_skip_threshold is not None:
            self.vision_skip_threshold = vision_skip_threshold
        if language_skip_threshold is not None:
            self.language_skip_threshold = language_skip_threshold
        if action_skip_threshold is not None:
            self.action_skip_threshold = action_skip_threshold

        raw_block_outputs = self._select_block_outputs(hidden_states)
        adapted_outputs, routing_weights = self._run_full_depth(raw_block_outputs)
        modality_masks = self._build_modality_masks(
            input_ids,
            image_token_ids=image_token_ids,
            action_token_id=action_token_id,
            action_token_range=action_token_range,
        )
        block_importance, modality_importance = self._compute_importance_stats(routing_weights, modality_masks)

        active_enable_skipping = self.enable_skipping if enable_skipping is None else enable_skipping
        active_skip_mode = self.skip_mode if skip_mode is None else skip_mode

        if active_enable_skipping and active_skip_mode != "none":
            keep_mask = self._build_keep_mask(
                block_importance=block_importance,
                modality_importance=modality_importance,
                modality_masks=modality_masks,
                skip_mode=active_skip_mode,
            )
            final_hidden, blocks_executed, executed_weights = self._rerun_with_keep_mask(raw_block_outputs, keep_mask)
        else:
            keep_mask = torch.ones(self.n_blocks, dtype=torch.bool, device=raw_block_outputs[0].device)
            final_hidden = adapted_outputs[-1]
            blocks_executed = [(block_idx, 0) for block_idx in range(self.n_blocks)]
            executed_weights = routing_weights

        result = {
            "hidden_states": final_hidden,
            "blocks_executed": blocks_executed,
            "num_blocks_executed": sum(1 for _, status in blocks_executed if status >= 0),
            "total_blocks": self.n_blocks,
            "flops_ratio": sum(1 for _, status in blocks_executed if status >= 0) / max(self.n_blocks, 1),
            "effective_block_ratio": sum(1 for _, status in blocks_executed if status >= 0) / max(self.n_blocks, 1),
            "backbone_compute_preserved": True,
            "block_importance": block_importance,
            "modality_importance": modality_importance,
            "keep_mask": keep_mask.tolist(),
            "skip_mode": active_skip_mode if active_enable_skipping else "none",
        }

        if return_routing_info:
            result["routing_weights"] = routing_weights
            result["executed_routing_weights"] = executed_weights

        return result


class StarVLABackboneSkipContext(AbstractContextManager):
    """
    Patch a Qwen language model so blocks can be skipped during a single forward.

    This is inference-only and assumes `use_cache=False`. The controller groups
    decoder layers into AttnRes blocks. At the first layer of each block, it
    computes a routed summary from previous block states, decides whether the
    block can be skipped, and either:

    1. executes all layers inside that block once, or
    2. bypasses them and reuses the routed summary.
    """

    def __init__(
        self,
        adapter: StarVLAAttnResAdapter,
        layer_modules: nn.ModuleList,
        *,
        layer_return_type: str,
        input_ids: Optional[torch.Tensor],
        image_token_ids: tuple[int, ...],
        action_token_id: Optional[int],
        action_token_range: Optional[tuple[int, int]],
        enable_skipping: bool,
        skip_mode: str,
        uniform_skip_threshold: Optional[float] = None,
        vision_skip_threshold: Optional[float] = None,
        language_skip_threshold: Optional[float] = None,
        action_skip_threshold: Optional[float] = None,
    ) -> None:
        self.adapter = adapter
        self.layer_modules = layer_modules
        self.layer_return_type = layer_return_type
        self.enable_skipping = enable_skipping
        self.skip_mode = skip_mode

        if uniform_skip_threshold is not None:
            self.adapter.uniform_skip_threshold = uniform_skip_threshold
        if vision_skip_threshold is not None:
            self.adapter.vision_skip_threshold = vision_skip_threshold
        if language_skip_threshold is not None:
            self.adapter.language_skip_threshold = language_skip_threshold
        if action_skip_threshold is not None:
            self.adapter.action_skip_threshold = action_skip_threshold

        self.modality_masks = self.adapter._build_modality_masks(
            input_ids,
            image_token_ids=image_token_ids,
            action_token_id=action_token_id,
            action_token_range=action_token_range,
        )
        self.block_layer_ranges = self.adapter.get_block_layer_ranges()
        self.block_start_to_idx = {start: idx for idx, (start, _) in enumerate(self.block_layer_ranges)}
        self.block_idx_to_layers = {
            idx: list(range(start, end + 1)) for idx, (start, end) in enumerate(self.block_layer_ranges)
        }
        self.original_forwards = [layer.forward for layer in layer_modules]
        self._patched = False
        self._passthrough_layers: set[int] = set()
        self.adapted_outputs: list[torch.Tensor] = []
        self.blocks_executed: list[tuple[int, int]] = []
        self.keep_mask: list[bool] = []
        self.block_scores: list[float] = []
        self.modality_scores: dict[str, list[float]] = {m: [] for m in self.adapter.MODALITIES}
        self.decision_metric = "mean_abs_delta"

    def _extract_hidden(self, layer_output: Any) -> torch.Tensor:
        if isinstance(layer_output, tuple):
            return layer_output[0]
        return layer_output

    def _format_return(self, hidden_states: torch.Tensor, kwargs: dict[str, Any]) -> Any:
        if self.layer_return_type == "tuple":
            if kwargs.get("output_attentions", False):
                return hidden_states, None
            return (hidden_states,)
        return hidden_states

    def _call_original(self, layer_idx: int, hidden_states: torch.Tensor, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        call_args = list(args)
        call_kwargs = dict(kwargs)
        if call_args:
            call_args[0] = hidden_states
            return self.original_forwards[layer_idx](*call_args, **call_kwargs)
        call_kwargs["hidden_states"] = hidden_states
        return self.original_forwards[layer_idx](**call_kwargs)

    def _run_or_skip_block(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if not self.adapted_outputs:
            self.adapted_outputs.append(hidden_states)

        routed_input, _ = self.adapter.block_attn_res[block_idx](self.adapted_outputs, return_weights=True)
        keep, block_score, modality_scores = self.adapter.should_keep_block_online(
            block_idx=block_idx,
            hidden_states=hidden_states,
            routed_input=routed_input,
            modality_masks=self.modality_masks,
            skip_mode=self.skip_mode if self.enable_skipping else "none",
        )

        self.keep_mask.append(keep)
        self.block_scores.append(block_score)
        for modality in self.adapter.MODALITIES:
            self.modality_scores[modality].append(modality_scores[modality])

        if keep:
            block_hidden = hidden_states
            for inner_layer_idx in self.block_idx_to_layers[block_idx]:
                layer_output = self._call_original(inner_layer_idx, block_hidden, args, kwargs)
                block_hidden = self._extract_hidden(layer_output)
            final_hidden = self.adapter.output_norm(block_hidden + routed_input)
            self.blocks_executed.append((block_idx, 0))
        else:
            final_hidden = self.adapter.output_norm(hidden_states + routed_input)
            self.blocks_executed.append((block_idx, -1))

        self.adapted_outputs.append(final_hidden)
        for inner_layer_idx in self.block_idx_to_layers[block_idx][1:]:
            self._passthrough_layers.add(inner_layer_idx)
        return final_hidden

    def _layer_forward(self, layer_idx: int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        hidden_states = args[0] if args else kwargs["hidden_states"]

        if layer_idx in self._passthrough_layers:
            self._passthrough_layers.remove(layer_idx)
            return self._format_return(hidden_states, kwargs)

        block_idx = self.block_start_to_idx.get(layer_idx)
        if block_idx is None:
            return self._call_original(layer_idx, hidden_states, args, kwargs)

        final_hidden = self._run_or_skip_block(block_idx, hidden_states, args, kwargs)
        return self._format_return(final_hidden, kwargs)

    def get_summary(self) -> dict[str, Any]:
        num_blocks_executed = sum(1 for _, status in self.blocks_executed if status >= 0)
        return {
            "blocks_executed": self.blocks_executed,
            "num_blocks_executed": num_blocks_executed,
            "total_blocks": self.adapter.n_blocks,
            "flops_ratio": num_blocks_executed / max(self.adapter.n_blocks, 1),
            "effective_block_ratio": num_blocks_executed / max(self.adapter.n_blocks, 1),
            "backbone_compute_preserved": False,
            "block_importance": self.block_scores,
            "modality_importance": self.modality_scores,
            "keep_mask": self.keep_mask,
            "skip_mode": self.skip_mode if self.enable_skipping else "none",
            "decision_metric": self.decision_metric,
        }

    def __enter__(self) -> "StarVLABackboneSkipContext":
        if self._patched:
            return self

        for layer_idx, layer in enumerate(self.layer_modules):
            def wrapped(layer_self, *args, __layer_idx=layer_idx, **kwargs):
                return self._layer_forward(__layer_idx, args, kwargs)

            layer.forward = MethodType(wrapped, layer)
        self._patched = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._patched:
            return
        for layer, original_forward in zip(self.layer_modules, self.original_forwards):
            layer.forward = original_forward
        self._patched = False


def build_starvla_attnres_adapter(config: Any, hidden_size: int, num_hidden_layers: int) -> Optional[StarVLAAttnResAdapter]:
    attnres_cfg = StarVLAAttnResConfig(
        enabled=bool(_get_cfg_value(config, "framework.attnres.enabled", False)),
        n_blocks=int(_get_cfg_value(config, "framework.attnres.n_blocks", 8)),
        temperature=float(_get_cfg_value(config, "framework.attnres.temperature", 1.0)),
        enable_skipping=bool(_get_cfg_value(config, "framework.attnres.enable_skipping", False)),
        skip_mode=str(_get_cfg_value(config, "framework.attnres.skip_mode", "none")),
        uniform_skip_threshold=float(_get_cfg_value(config, "framework.attnres.uniform_skip_threshold", 0.01)),
        vision_skip_threshold=float(_get_cfg_value(config, "framework.attnres.vision_skip_threshold", 0.02)),
        language_skip_threshold=float(_get_cfg_value(config, "framework.attnres.language_skip_threshold", 0.01)),
        action_skip_threshold=float(_get_cfg_value(config, "framework.attnres.action_skip_threshold", 0.005)),
    )
    if not attnres_cfg.enabled:
        return None
    return StarVLAAttnResAdapter(
        d_model=hidden_size,
        num_hidden_layers=num_hidden_layers,
        config=attnres_cfg,
    )
