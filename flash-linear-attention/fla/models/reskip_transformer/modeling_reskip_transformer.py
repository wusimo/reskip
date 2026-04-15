from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.models.reskip_transformer.configuration_reskip_transformer import ReSkipTransformerConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as TransformerMLP
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


def _is_dtensor(value: Any) -> bool:
    return type(value).__name__ == "DTensor"


def _materialize_if_dtensor(value: torch.Tensor) -> torch.Tensor:
    if _is_dtensor(value):
        return value.full_tensor()
    return value


def blend_states(
    old_states: torch.Tensor,
    new_states: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    if active_mask is None:
        return new_states
    mask = active_mask[:, None, None].to(dtype=new_states.dtype)
    return new_states * mask + old_states * (1.0 - mask)


def _router_effective_query(router: "BlockAttentionResidual") -> torch.Tensor:
    query = _materialize_if_dtensor(router.w_query).float()
    norm_weight = getattr(router.key_norm, "weight", None)
    if norm_weight is not None:
        query = query * _materialize_if_dtensor(norm_weight).float()
    return query


def _rms_norm_base(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    hidden_states_fp32 = hidden_states.float()
    inv_rms = torch.rsqrt(hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return hidden_states_fp32 * inv_rms


def batch_attend_completed_blocks(
    routers: list["BlockAttentionResidual"],
    completed_blocks: list[torch.Tensor],
    return_weights: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]]:
    if any(_is_dtensor(_router_effective_query(router)) for router in routers):
        return [
            attend_completed_blocks(router, completed_blocks, return_weights)
            for router in routers
        ]

    values = torch.stack(completed_blocks, dim=0)
    base_keys = _rms_norm_base(values, routers[0].key_norm.eps)
    queries = torch.stack([_router_effective_query(router) for router in routers], dim=0)
    scale = math.sqrt(values.shape[-1]) * routers[0].temperature
    scores = torch.einsum("qd,nbtd->qnbt", queries, base_keys) / scale
    max_scores = scores.amax(dim=1)
    exp_scores = torch.exp(scores - max_scores.unsqueeze(1))
    lse = exp_scores.sum(dim=1)
    outputs = torch.einsum("qnbt,nbtd->qbtd", exp_scores, values.float()).to(values.dtype)
    probs = exp_scores / lse.unsqueeze(1)
    log_probs = scores - max_scores.unsqueeze(1) - torch.log(lse).unsqueeze(1)
    entropies = -(probs * log_probs).sum(dim=1)

    if return_weights:
        weights = probs.permute(0, 2, 3, 1).to(values.dtype)
    else:
        weights = None

    result = []
    for idx in range(len(routers)):
        result.append(
            (
                outputs[idx],
                max_scores[idx],
                lse[idx],
                entropies[idx],
                None if weights is None else weights[idx],
                probs[idx].mean(dim=(1, 2)).to(values.dtype),
            )
        )
    return result


def attend_completed_blocks(
    router: "BlockAttentionResidual",
    completed_blocks: list[torch.Tensor],
    return_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if not completed_blocks:
        raise ValueError("Expected at least one completed block.")

    if len(completed_blocks) == 1:
        routed = completed_blocks[0]
        phase1_max = routed.new_zeros(routed.shape[0], routed.shape[1])
        phase1_lse = routed.new_ones(routed.shape[0], routed.shape[1])
        phase1_entropy = routed.new_zeros(routed.shape[0], routed.shape[1])
        weights = routed.new_ones(routed.shape[0], routed.shape[1], 1) if return_weights else None
        source_means = routed.new_ones(1)
        return routed, phase1_max, phase1_lse, phase1_entropy, weights, source_means

    sources = torch.stack(completed_blocks, dim=2)
    base_keys = _rms_norm_base(sources, router.key_norm.eps)
    scores = torch.einsum("d,btnd->btn", _router_effective_query(router), base_keys)
    scores = scores / (math.sqrt(sources.shape[-1]) * router.temperature)
    phase1_max = scores.amax(dim=-1)
    shifted_scores = scores - phase1_max.unsqueeze(-1)
    exp_scores = torch.exp(shifted_scores)
    phase1_lse = exp_scores.sum(dim=-1)
    routed = torch.sum(exp_scores.unsqueeze(-1) * sources.float(), dim=2).to(sources.dtype)
    probs = exp_scores / phase1_lse.unsqueeze(-1)
    log_probs = shifted_scores - torch.log(phase1_lse).unsqueeze(-1)
    phase1_entropy = -(probs * log_probs).sum(dim=-1)
    weights = probs.to(sources.dtype) if return_weights else None
    source_means = probs.mean(dim=(0, 1)).to(sources.dtype)
    return routed, phase1_max, phase1_lse, phase1_entropy, weights, source_means


def merge_with_partial_block(
    router: "BlockAttentionResidual",
    phase1: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor],
    partial_block: torch.Tensor | None,
    return_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    phase1_output, phase1_max, phase1_lse, phase1_entropy, phase1_weights, _phase1_source_means = phase1
    if partial_block is None:
        hidden_states = phase1_output * phase1_lse.reciprocal().to(phase1_output.dtype).unsqueeze(-1)
        return hidden_states, phase1_entropy, phase1_weights

    partial_scores = torch.einsum(
        "d,btd->bt",
        _router_effective_query(router),
        _rms_norm_base(partial_block, router.key_norm.eps),
    ) / (math.sqrt(partial_block.shape[-1]) * router.temperature)
    merged_max = torch.maximum(phase1_max, partial_scores)
    phase1_coeff = torch.exp(phase1_max - merged_max)
    partial_coeff = torch.exp(partial_scores - merged_max)
    denom = phase1_coeff * phase1_lse + partial_coeff
    phase1_weight = (phase1_coeff * phase1_lse / denom).to(partial_block.dtype)
    partial_weight = (partial_coeff / denom).to(partial_block.dtype)
    hidden_states = (
        phase1_output * phase1_weight.unsqueeze(-1)
        + partial_block * partial_weight.unsqueeze(-1)
    )
    phase1_weight_fp32 = phase1_weight.float().clamp_min(1e-8)
    partial_weight_fp32 = partial_weight.float().clamp_min(1e-8)
    entropy = (
        phase1_weight_fp32 * phase1_entropy.float()
        - phase1_weight_fp32 * torch.log(phase1_weight_fp32)
        - partial_weight_fp32 * torch.log(partial_weight_fp32)
    ).to(hidden_states.dtype)

    if not return_weights:
        return hidden_states, entropy, None

    if phase1_weights is None:
        raise RuntimeError("Phase-1 weights are required when return_weights=True.")
    weights = torch.cat(
        [phase1_weights * phase1_weight.unsqueeze(-1), partial_weight.unsqueeze(-1)],
        dim=-1,
    )
    return hidden_states, entropy, weights


def normalize_router_entropy(entropy: torch.Tensor, num_sources: int) -> torch.Tensor:
    if num_sources <= 1:
        return torch.zeros_like(entropy)
    max_entropy = math.log(float(num_sources))
    return (entropy.float() / max_entropy).clamp_(0.0, 1.0).to(entropy.dtype)


def collect_completed_blocks(
    block_states: list[torch.Tensor | None],
    current_block_idx: int,
) -> tuple[list[torch.Tensor], list[int]]:
    source_states = [block_states[0]]
    source_ids = [-1]
    for block_idx in range(current_block_idx):
        state = block_states[block_idx + 1]
        if state is not None:
            source_states.append(state)
            source_ids.append(block_idx)
    return source_states, source_ids


def summarize_phase1_output(
    phase1_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor],
    source_ids: list[int],
) -> dict[str, float | None]:
    _output, _max, _lse, entropy, _weights, source_means = phase1_output
    avg_entropy = float(normalize_router_entropy(entropy, len(source_ids)).mean().item())
    avg_embed = float(source_means[0].float().item()) if source_means.numel() > 0 else None
    avg_recent = float(source_means[-1].float().item()) if len(source_ids) > 1 else None
    return {
        "avg_phase1_entropy": avg_entropy,
        "avg_phase1_embed_weight": avg_embed,
        "avg_phase1_recent_weight": avg_recent,
        "num_completed_sources": float(len(source_ids)),
    }


def dynamic_skip_required_features(strategy: str | None) -> tuple[bool, bool, bool]:
    strategy = normalize_dynamic_skip_strategy(strategy)
    if strategy is None:
        return False, False, False
    if strategy in {"recent_weight_lt", "recent_weight_gt"}:
        return True, False, False
    if strategy == "embed_weight_gt":
        return False, True, False
    if strategy == "entropy_lt":
        return False, False, True
    if strategy in {"recent_minus_embed_gt", "recent_over_embed_gt"}:
        return True, True, False
    if strategy in {"recent_confidence_gt", "recent_x_entropy_lt"}:
        return True, False, True
    if strategy == "recent_margin_confidence_gt":
        return True, True, True
    raise ValueError(f"Unsupported dynamic skip strategy: {strategy}")


def summarize_probe_only_phase1(
    routers: list["BlockAttentionResidual"],
    completed_blocks: list[torch.Tensor],
    source_ids: list[int],
    *,
    strategy: str | None,
) -> dict[str, float | None]:
    need_recent, need_embed, need_entropy = dynamic_skip_required_features(strategy)

    if not completed_blocks:
        raise ValueError("Expected at least one completed block.")
    if len(completed_blocks) == 1:
        return {
            "avg_phase1_entropy": 0.0 if need_entropy else None,
            "avg_phase1_embed_weight": 1.0 if need_embed else None,
            "avg_phase1_recent_weight": None,
            "num_completed_sources": float(len(source_ids)),
        }

    if any(_is_dtensor(_router_effective_query(router)) for router in routers):
        phase1_outputs = batch_attend_completed_blocks(
            routers=routers,
            completed_blocks=completed_blocks,
            return_weights=False,
        )
        per_router = [summarize_phase1_output(item, source_ids) for item in phase1_outputs]
        avg_entropy = (
            sum(item["avg_phase1_entropy"] for item in per_router if item["avg_phase1_entropy"] is not None)
            / max(sum(1 for item in per_router if item["avg_phase1_entropy"] is not None), 1)
            if need_entropy
            else None
        )
        avg_embed = (
            sum(item["avg_phase1_embed_weight"] for item in per_router if item["avg_phase1_embed_weight"] is not None)
            / max(sum(1 for item in per_router if item["avg_phase1_embed_weight"] is not None), 1)
            if need_embed
            else None
        )
        avg_recent = (
            sum(item["avg_phase1_recent_weight"] for item in per_router if item["avg_phase1_recent_weight"] is not None)
            / max(sum(1 for item in per_router if item["avg_phase1_recent_weight"] is not None), 1)
            if need_recent
            else None
        )
        return {
            "avg_phase1_entropy": avg_entropy,
            "avg_phase1_embed_weight": avg_embed,
            "avg_phase1_recent_weight": avg_recent,
            "num_completed_sources": float(len(source_ids)),
        }

    values = torch.stack(completed_blocks, dim=0)
    base_keys = _rms_norm_base(values, routers[0].key_norm.eps)
    queries = torch.stack([_router_effective_query(router) for router in routers], dim=0)
    scale = math.sqrt(values.shape[-1]) * routers[0].temperature
    scores = torch.einsum("qd,nbtd->qnbt", queries, base_keys) / scale
    probs = torch.softmax(scores, dim=1)
    source_means = probs.mean(dim=(2, 3))

    avg_embed = float(source_means[:, 0].float().mean().item()) if need_embed else None
    avg_recent = float(source_means[:, -1].float().mean().item()) if need_recent else None
    avg_entropy = None
    if need_entropy:
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum(dim=1)
        avg_entropy = float(normalize_router_entropy(entropy, len(source_ids)).float().mean().item())
    return {
        "avg_phase1_entropy": avg_entropy,
        "avg_phase1_embed_weight": avg_embed,
        "avg_phase1_recent_weight": avg_recent,
        "num_completed_sources": float(len(source_ids)),
    }

def normalize_dynamic_skip_strategy(strategy: str | None) -> str | None:
    if strategy is None:
        return None
    if strategy.startswith("prev_"):
        return strategy[5:]
    return strategy


def is_cached_prev_dynamic_strategy(strategy: str | None) -> bool:
    return strategy is not None and strategy.startswith("prev_")


def should_skip_dynamic_position(
    *,
    strategy: str | None,
    threshold: float | None,
    position_thresholds: list[float] | None,
    phase1_stats: dict[str, float | None],
    position: int,
    num_positions: int,
    skipped_so_far: int,
    max_skips: int | None,
) -> bool:
    if strategy is None:
        return False
    strategy = normalize_dynamic_skip_strategy(strategy)
    if threshold is None and position_thresholds is None:
        return False
    if position <= 0 or position >= num_positions - 1:
        return False
    if max_skips is not None and skipped_so_far >= max_skips:
        return False
    effective_threshold = threshold
    if position_thresholds is not None:
        if position >= len(position_thresholds):
            raise ValueError(
                f"dynamic_skip_position_thresholds length {len(position_thresholds)} is too short for position {position}."
            )
        effective_threshold = position_thresholds[position]
    if effective_threshold is None:
        return False
    recent_weight = phase1_stats.get("avg_phase1_recent_weight")
    embed_weight = phase1_stats.get("avg_phase1_embed_weight")
    entropy = phase1_stats.get("avg_phase1_entropy")
    if strategy == "recent_weight_lt":
        return recent_weight is not None and recent_weight < effective_threshold
    if strategy == "recent_weight_gt":
        return recent_weight is not None and recent_weight > effective_threshold
    if strategy == "embed_weight_gt":
        return embed_weight is not None and embed_weight > effective_threshold
    if strategy == "entropy_lt":
        return entropy is not None and entropy < effective_threshold
    if strategy == "recent_minus_embed_gt":
        return (
            recent_weight is not None
            and embed_weight is not None
            and (recent_weight - embed_weight) > effective_threshold
        )
    if strategy == "recent_over_embed_gt":
        return (
            recent_weight is not None
            and embed_weight is not None
            and embed_weight > 1e-8
            and (recent_weight / embed_weight) > effective_threshold
        )
    if strategy == "recent_confidence_gt":
        return (
            recent_weight is not None
            and entropy is not None
            and (recent_weight * max(1.0 - entropy, 0.0)) > effective_threshold
        )
    if strategy == "recent_margin_confidence_gt":
        return (
            recent_weight is not None
            and embed_weight is not None
            and entropy is not None
            and ((recent_weight - embed_weight) * max(1.0 - entropy, 0.0)) > effective_threshold
        )
    if strategy == "recent_x_entropy_lt":
        return (
            recent_weight is not None
            and entropy is not None
            and (recent_weight * max(1.0 - entropy, 0.0)) < effective_threshold
        )
    raise ValueError(f"Unsupported dynamic skip strategy: {strategy}")


def dynamic_skip_needs_phase1_position(
    *,
    strategy: str | None,
    threshold: float | None,
    position_thresholds: list[float] | None,
    position: int,
    num_positions: int,
    skipped_so_far: int,
    max_skips: int | None,
) -> bool:
    if strategy is None:
        return False
    if is_cached_prev_dynamic_strategy(strategy):
        return False
    if threshold is None and position_thresholds is None:
        return False
    if position <= 0 or position >= num_positions - 1:
        return False
    if max_skips is not None and skipped_so_far >= max_skips:
        return False
    effective_threshold = threshold
    if position_thresholds is not None:
        if position >= len(position_thresholds):
            raise ValueError(
                f"dynamic_skip_position_thresholds length {len(position_thresholds)} is too short for position {position}."
            )
        effective_threshold = position_thresholds[position]
    if effective_threshold is None:
        return False
    if abs(float(effective_threshold)) >= 1e8:
        return False
    return True


@dataclass
class ReSkipBaseModelOutputWithPast(BaseModelOutputWithPast):
    routing_info: dict[str, Any] | None = None
    routing_entropy_tensor: torch.Tensor | None = None


@dataclass
class ReSkipCausalLMOutputWithPast(CausalLMOutputWithPast):
    routing_info: dict[str, Any] | None = None
    routing_entropy_tensor: torch.Tensor | None = None


class BlockAttentionResidual(nn.Module):
    """Block AttnRes over completed blocks plus the current partial block."""

    def __init__(self, config: ReSkipTransformerConfig):
        super().__init__()
        self.temperature = config.attn_res_temperature
        self.w_query = nn.Parameter(torch.empty(config.hidden_size))
        self.key_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=config.elementwise_affine,
        )

    def reset_parameters(self, initializer_range: float) -> None:
        # Match the paper's pseudo-query initialization: near-uniform routing
        # with small random asymmetry so routers can specialize early.
        nn.init.normal_(self.w_query, mean=0.0, std=initializer_range)
        if getattr(self.key_norm, "weight", None) is not None:
            nn.init.ones_(self.key_norm.weight)

    def forward(
        self,
        source_states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not source_states:
            raise ValueError("BlockAttentionResidual requires at least one source state.")

        if len(source_states) == 1:
            routed = source_states[0]
            weights = routed.new_ones(routed.shape[0], routed.shape[1], 1)
            return routed, weights if return_weights else None

        sources = torch.stack(source_states, dim=2)
        base_keys = _rms_norm_base(sources, self.key_norm.eps)
        scores = torch.einsum("d,btnd->btn", _router_effective_query(self), base_keys)
        scores = scores / (math.sqrt(sources.shape[-1]) * self.temperature)
        weights = torch.softmax(scores, dim=-1).to(sources.dtype)
        routed = torch.sum(weights.unsqueeze(-1) * sources, dim=2).contiguous()
        return routed, weights if return_weights else None


class ReSkipTransformerLayer(GradientCheckpointingLayer):
    """Standard transformer layer split into attention and MLP phases."""

    def __init__(self, config: ReSkipTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_router = BlockAttentionResidual(config)
        self.mlp_router = BlockAttentionResidual(config)
        # Keep FLA's RMSNorm for the standard 3D transformer path because flame's
        # sequence-parallel / DTensor stack expects norm modules that understand DTensor.
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.Tensor, Cache | None]:
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, _, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
            cache_layer_idx=cache_layer_idx,
            **kwargs,
        )
        return hidden_states, past_key_values

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        return hidden_states

    def forward(
        self,
        block_input: torch.Tensor,
        partial_block: torch.Tensor | None,
        attn_phase1: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
        mlp_phase1: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        active_mask: torch.Tensor | None = None,
        return_routing_weights: bool = False,
        skip_mlp: bool = False,
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple[
        torch.Tensor,
        Cache | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        attn_input, attn_entropy, attn_weights = merge_with_partial_block(
            self.attn_router,
            attn_phase1,
            partial_block,
            return_weights=return_routing_weights,
        )
        attn_out, past_key_values = self.forward_attention(
            attn_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_layer_idx=cache_layer_idx,
            **kwargs,
        )
        old_partial = partial_block
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        partial_block = blend_states(
            block_input if old_partial is None else old_partial,
            partial_block,
            active_mask,
        )

        if skip_mlp:
            _phase1_output, _phase1_max, _phase1_lse, mlp_entropy, _phase1_weights, _source_means = mlp_phase1
            mlp_weights = None
        else:
            mlp_input, mlp_entropy, mlp_weights = merge_with_partial_block(
                self.mlp_router,
                mlp_phase1,
                partial_block,
                return_weights=return_routing_weights,
            )
            mlp_out = self.forward_mlp(mlp_input, **kwargs)
            old_partial = partial_block
            partial_block = partial_block + mlp_out
            partial_block = blend_states(old_partial, partial_block, active_mask)

        return partial_block, past_key_values, attn_weights, mlp_weights, attn_entropy, mlp_entropy, skip_mlp


class ReSkipBlockGroup(nn.Module):
    def __init__(
        self,
        config: ReSkipTransformerConfig,
        block_idx: int,
        layers_per_block: int,
        first_layer_idx: int,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.layers = nn.ModuleList(
            [ReSkipTransformerLayer(config, first_layer_idx + offset) for offset in range(layers_per_block)]
        )

    def _select_probe_routers(self, probe_mode: str) -> list["BlockAttentionResidual"]:
        if probe_mode == "all":
            routers = []
            for layer in self.layers:
                routers.extend([layer.attn_router, layer.mlp_router])
            return routers
        if probe_mode == "attn_only":
            return [layer.attn_router for layer in self.layers]
        if probe_mode == "first_layer":
            layer = self.layers[0]
            return [layer.attn_router, layer.mlp_router]
        if probe_mode == "first_attn":
            return [self.layers[0].attn_router]
        raise ValueError(f"Unsupported dynamic skip probe mode: {probe_mode}")

    def prepare_phase1(
        self,
        block_states: list[torch.Tensor | None],
        current_block_idx: int,
        return_routing_weights: bool = False,
        probe_mode: str = "all",
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]],
        list[int],
    ]:
        completed_blocks, completed_source_ids = collect_completed_blocks(block_states, current_block_idx)
        routers = self._select_probe_routers(probe_mode)
        phase1_outputs = batch_attend_completed_blocks(
            routers=routers,
            completed_blocks=completed_blocks,
            return_weights=return_routing_weights,
        )
        return phase1_outputs, completed_source_ids

    def probe_phase1_stats(
        self,
        block_states: list[torch.Tensor | None],
        current_block_idx: int,
        *,
        probe_mode: str,
        strategy: str | None,
    ) -> dict[str, float | None]:
        completed_blocks, completed_source_ids = collect_completed_blocks(block_states, current_block_idx)
        routers = self._select_probe_routers(probe_mode)
        return summarize_probe_only_phase1(
            routers=routers,
            completed_blocks=completed_blocks,
            source_ids=completed_source_ids,
            strategy=strategy,
        )

    def summarize_phase1(
        self,
        phase1_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]],
        completed_source_ids: list[int],
    ) -> dict[str, float | None]:
        per_router = [summarize_phase1_output(item, completed_source_ids) for item in phase1_outputs]
        avg_entropy = (
            sum(item["avg_phase1_entropy"] for item in per_router if item["avg_phase1_entropy"] is not None)
            / max(sum(1 for item in per_router if item["avg_phase1_entropy"] is not None), 1)
            if per_router
            else None
        )
        avg_embed = (
            sum(item["avg_phase1_embed_weight"] for item in per_router if item["avg_phase1_embed_weight"] is not None)
            / max(sum(1 for item in per_router if item["avg_phase1_embed_weight"] is not None), 1)
            if per_router
            else None
        )
        avg_recent = (
            sum(item["avg_phase1_recent_weight"] for item in per_router if item["avg_phase1_recent_weight"] is not None)
            / max(sum(1 for item in per_router if item["avg_phase1_recent_weight"] is not None), 1)
            if per_router
            else None
        )
        return {
            "avg_phase1_entropy": avg_entropy,
            "avg_phase1_embed_weight": avg_embed,
            "avg_phase1_recent_weight": avg_recent,
            "num_completed_sources": float(len(completed_source_ids)),
        }

    def forward(
        self,
        block_states: list[torch.Tensor | None],
        current_block_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        active_mask: torch.Tensor | None = None,
        return_routing_weights: bool = False,
        phase1_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]] | None = None,
        completed_source_ids: list[int] | None = None,
        cache_layer_offset: int | None = None,
        dynamic_skip_granularity: str = "block",
        dynamic_skip_strategy: str | None = None,
        dynamic_skip_threshold: float | None = None,
        dynamic_skip_position_thresholds: list[float] | None = None,
        dynamic_skip_max_skips: int | None = None,
        dynamic_skips_taken: int = 0,
        collect_dynamic_skip_stats: bool = False,
        **kwargs: Unpack[Any],
    ) -> tuple[
        torch.Tensor,
        Cache | None,
        list[tuple[list[int], torch.Tensor | None]],
        list[tuple[list[int], torch.Tensor | None]],
        list[torch.Tensor],
        dict[str, float | None] | None,
        list[dict[str, Any]],
        list[dict[str, Any]],
        int,
        int,
    ]:
        block_input = block_states[current_block_idx]
        if block_input is None:
            raise RuntimeError("Expected the current block input to be available in block_states.")

        if phase1_outputs is None or completed_source_ids is None:
            phase1_outputs, completed_source_ids = self.prepare_phase1(
                block_states=block_states,
                current_block_idx=current_block_idx,
                return_routing_weights=return_routing_weights,
            )

        partial_block = None
        next_cache = past_key_values
        attn_records: list[tuple[list[int], torch.Tensor | None]] = []
        mlp_records: list[tuple[list[int], torch.Tensor | None]] = []
        need_router_entropy = self.training or collect_dynamic_skip_stats
        router_entropies: list[torch.Tensor] = []
        layer_execution_trace: list[dict[str, Any]] = []
        mlp_execution_trace: list[dict[str, Any]] = []
        skip_units_taken = dynamic_skips_taken
        compute_units_executed = 0
        block_phase1_summary = None
        if dynamic_skip_granularity == "block" and (
            collect_dynamic_skip_stats or is_cached_prev_dynamic_strategy(dynamic_skip_strategy)
        ):
            block_phase1_summary = self.summarize_phase1(phase1_outputs, completed_source_ids)

        for layer_idx, layer in enumerate(self.layers):
            attn_phase1 = phase1_outputs[2 * layer_idx]
            mlp_phase1 = phase1_outputs[2 * layer_idx + 1]
            attn_source_ids = list(completed_source_ids) if partial_block is None else [*completed_source_ids, current_block_idx]
            cache_layer_idx = None if cache_layer_offset is None else cache_layer_offset + layer_idx
            global_layer_idx = layer.layer_idx
            mlp_phase1_stats = None
            skip_mlp = False
            mlp_need_position = (
                dynamic_skip_granularity == "mlp"
                and dynamic_skip_needs_phase1_position(
                    strategy=dynamic_skip_strategy,
                    threshold=dynamic_skip_threshold,
                    position_thresholds=dynamic_skip_position_thresholds,
                    position=global_layer_idx,
                    num_positions=self.layers[0].config.num_hidden_layers,
                    skipped_so_far=skip_units_taken,
                    max_skips=dynamic_skip_max_skips,
                )
            )
            if dynamic_skip_granularity == "mlp" and (mlp_need_position or collect_dynamic_skip_stats):
                mlp_phase1_stats = summarize_phase1_output(mlp_phase1, completed_source_ids)
            if mlp_need_position:
                skip_mlp = should_skip_dynamic_position(
                    strategy=dynamic_skip_strategy,
                    threshold=dynamic_skip_threshold,
                    position_thresholds=dynamic_skip_position_thresholds,
                    phase1_stats=mlp_phase1_stats if mlp_phase1_stats is not None else {},
                    position=global_layer_idx,
                    num_positions=self.layers[0].config.num_hidden_layers,
                    skipped_so_far=skip_units_taken,
                    max_skips=dynamic_skip_max_skips,
                )
            partial_block, next_cache, attn_weights, mlp_weights, attn_entropy, mlp_entropy, mlp_was_skipped = layer(
                block_input=block_input,
                partial_block=partial_block,
                attn_phase1=attn_phase1,
                mlp_phase1=mlp_phase1,
                attention_mask=attention_mask,
                past_key_values=next_cache,
                use_cache=use_cache,
                active_mask=active_mask,
                return_routing_weights=return_routing_weights,
                skip_mlp=skip_mlp,
                cache_layer_idx=cache_layer_idx,
                **kwargs,
            )
            attn_records.append((attn_source_ids, attn_weights))
            mlp_records.append(([*completed_source_ids, current_block_idx], mlp_weights))
            compute_units_executed += 1
            if mlp_was_skipped:
                skip_units_taken += 1
            else:
                compute_units_executed += 1
            attn_num_sources = len(attn_source_ids)
            mlp_num_sources = len(completed_source_ids) + 1
            if need_router_entropy:
                router_entropies.extend(
                    [
                        normalize_router_entropy(attn_entropy, attn_num_sources).mean(),
                        normalize_router_entropy(mlp_entropy, mlp_num_sources).mean(),
                    ]
                )
            if dynamic_skip_granularity == "mlp" and collect_dynamic_skip_stats:
                payload = {
                    "position": global_layer_idx,
                    "block_position": current_block_idx,
                    "block_idx": self.block_idx,
                    "local_layer_idx": layer_idx,
                    "status": "skipped" if mlp_was_skipped else "executed",
                    "executed_fraction": 0.0 if mlp_was_skipped else 1.0,
                }
                if mlp_phase1_stats is not None:
                    payload.update(mlp_phase1_stats)
                mlp_execution_trace.append(payload)

        return (
            partial_block,
            next_cache,
            attn_records,
            mlp_records,
            router_entropies,
            block_phase1_summary,
            layer_execution_trace,
            mlp_execution_trace,
            skip_units_taken,
            compute_units_executed,
        )


class ReSkipTransformerPreTrainedModel(PreTrainedModel):
    config_class = ReSkipTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReSkipBlockGroup", "ReSkipTransformerLayer"]
    _supports_cache_class = True
    _keys_to_ignore_on_load_unexpected = [
        r"model\.halt_norm\..*",
        r"model\.halt_head\..*",
        r"model\.halt_phase1_proj\..*",
        r"model\.halt_position_bias",
    ]

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BlockAttentionResidual):
            module.reset_parameters(self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if rescale_prenorm_residual:
            weight = None
            if hasattr(module, "o_proj"):
                weight = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                weight = module.down_proj.weight
            if weight is not None:
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                with torch.no_grad():
                    weight /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class ReSkipTransformerModel(ReSkipTransformerPreTrainedModel):
    def __init__(self, config: ReSkipTransformerConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers_per_block = config.num_hidden_layers // config.attn_res_num_blocks
        self.num_block_positions = config.attn_res_num_blocks
        self.num_unique_blocks = config.attn_res_num_blocks
        self.block_schedule = self._build_block_schedule()

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                ReSkipBlockGroup(
                    config=config,
                    block_idx=block_idx,
                    layers_per_block=self.layers_per_block,
                    first_layer_idx=block_idx * self.layers_per_block,
                )
                for block_idx in range(self.num_unique_blocks)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self._skip_keep_mask = self._normalize_keep_mask(config.skip_keep_mask)
        self._dynamic_skip_position_thresholds = self._normalize_dynamic_skip_position_thresholds(
            config.dynamic_skip_position_thresholds,
            getattr(config, "dynamic_skip_granularity", "block"),
        )
        self._last_routing_info: dict[str, Any] | None = None

        self.post_init()

    def _build_block_schedule(self) -> list[int]:
        return list(range(self.config.attn_res_num_blocks))

    def _should_skip_dynamic(
        self,
        strategy: str | None,
        threshold: float | None,
        position_thresholds: list[float] | None,
        phase1_stats: dict[str, float | None],
        position: int,
        skipped_so_far: int,
        max_skips: int | None,
    ) -> bool:
        return should_skip_dynamic_position(
            strategy=strategy,
            threshold=threshold,
            position_thresholds=position_thresholds,
            phase1_stats=phase1_stats,
            position=position,
            num_positions=self.num_block_positions,
            skipped_so_far=skipped_so_far,
            max_skips=max_skips,
        )

    def _dynamic_skip_needs_phase1(
        self,
        strategy: str | None,
        threshold: float | None,
        position_thresholds: list[float] | None,
        position: int,
        skipped_so_far: int,
        max_skips: int | None,
    ) -> bool:
        return dynamic_skip_needs_phase1_position(
            strategy=strategy,
            threshold=threshold,
            position_thresholds=position_thresholds,
            position=position,
            num_positions=self.num_block_positions,
            skipped_so_far=skipped_so_far,
            max_skips=max_skips,
        )

    def _normalize_keep_mask(
        self,
        keep_mask: list[int] | list[bool] | torch.Tensor | None,
    ) -> list[bool] | None:
        if keep_mask is None:
            return None
        if isinstance(keep_mask, torch.Tensor):
            keep_mask = keep_mask.tolist()
        keep_mask = [bool(value) for value in keep_mask]
        if len(keep_mask) != self.num_block_positions:
            raise ValueError(
                f"Expected keep mask of length {self.num_block_positions}, got {len(keep_mask)}."
            )
        return keep_mask

    def set_skip_keep_mask(self, keep_mask: list[int] | list[bool] | torch.Tensor | None) -> None:
        normalized = self._normalize_keep_mask(keep_mask)
        self._skip_keep_mask = normalized
        self.config.skip_keep_mask = normalized
        self.config.enable_skip_inference = normalized is not None

    def clear_skip_keep_mask(self) -> None:
        self._skip_keep_mask = None
        self.config.skip_keep_mask = None
        self.config.enable_skip_inference = False

    def _normalize_dynamic_skip_position_thresholds(
        self,
        thresholds: list[float] | torch.Tensor | None,
        granularity: str | None = None,
    ) -> list[float] | None:
        if thresholds is None:
            return None
        granularity = granularity or getattr(self.config, "dynamic_skip_granularity", "block")
        if isinstance(thresholds, torch.Tensor):
            thresholds = thresholds.tolist()
        thresholds = [float(value) for value in thresholds]
        expected_positions = self.num_block_positions if granularity == "block" else self.config.num_hidden_layers
        if len(thresholds) != expected_positions:
            raise ValueError(
                f"Expected dynamic position thresholds of length {expected_positions} for "
                f"granularity={granularity}, got {len(thresholds)}."
            )
        return thresholds

    def set_dynamic_skip_policy(
        self,
        *,
        strategy: str,
        granularity: str | None = None,
        probe_mode: str | None = None,
        threshold: float | None = None,
        position_thresholds: list[float] | torch.Tensor | None = None,
        max_skips: int | None = None,
    ) -> None:
        granularity = granularity or getattr(self.config, "dynamic_skip_granularity", "block")
        normalized_thresholds = self._normalize_dynamic_skip_position_thresholds(position_thresholds, granularity)
        self.config.dynamic_skip_strategy = strategy
        self.config.dynamic_skip_granularity = granularity
        self.config.dynamic_skip_probe_mode = probe_mode if probe_mode is not None else "all"
        self.config.dynamic_skip_threshold = None if threshold is None else float(threshold)
        self._dynamic_skip_position_thresholds = normalized_thresholds
        self.config.dynamic_skip_position_thresholds = normalized_thresholds
        self.config.dynamic_skip_max_skips = None if max_skips is None else int(max_skips)

    def clear_dynamic_skip_policy(self) -> None:
        self.config.dynamic_skip_strategy = None
        self.config.dynamic_skip_granularity = "block"
        self.config.dynamic_skip_probe_mode = "all"
        self.config.dynamic_skip_threshold = None
        self._dynamic_skip_position_thresholds = None
        self.config.dynamic_skip_position_thresholds = None
        self.config.dynamic_skip_max_skips = None

    def _resolve_keep_mask(
        self,
        enable_skipping: bool | None,
        skip_keep_mask: list[int] | list[bool] | torch.Tensor | None,
    ) -> list[bool] | None:
        runtime_mask = self._normalize_keep_mask(skip_keep_mask)
        configured_mask = runtime_mask if runtime_mask is not None else self._skip_keep_mask
        if enable_skipping is None:
            active = self.config.enable_skip_inference and configured_mask is not None
        else:
            active = enable_skipping and configured_mask is not None
        return configured_mask if active else None

    @staticmethod
    def build_keep_mask_from_importance(
        block_importance: list[float],
        threshold: float,
    ) -> list[bool]:
        keep_mask = [score >= threshold for score in block_importance]
        if keep_mask:
            keep_mask[0] = True
            keep_mask[-1] = True
        return keep_mask

    def _record_routing(
        self,
        storage: list[dict[str, Any]],
        block_idx: int,
        site: str,
        source_ids: list[int],
        weights: torch.Tensor,
    ) -> None:
        storage.append(
            {
                "target_block": block_idx,
                "site": site,
                "source_ids": list(source_ids),
                "weights": weights.detach(),
            }
        )

    def _aggregate_routing(
        self,
        routing_events: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[float], list[float]]:
        device = self.embeddings.weight.device
        matrix = torch.zeros(
            self.num_block_positions,
            self.num_block_positions,
            device=device,
            dtype=torch.float32,
        )
        self_importance = torch.zeros(self.num_block_positions, device=device, dtype=torch.float32)

        for event in routing_events:
            avg_weights = event["weights"].mean(dim=(0, 1)).float()
            target_block = int(event["target_block"])
            for source_weight, source_id in zip(avg_weights, event["source_ids"], strict=True):
                if source_id < 0:
                    continue
                if source_id == target_block:
                    self_importance[target_block] = torch.maximum(self_importance[target_block], source_weight)
                elif source_id < target_block:
                    matrix[source_id, target_block] = torch.maximum(matrix[source_id, target_block], source_weight)

        block_importance = []
        for block_idx in range(self.num_block_positions):
            downstream = matrix[block_idx, block_idx + 1 :]
            block_importance.append(float(downstream.max().item()) if downstream.numel() > 0 else 1.0)

        return matrix, block_importance, self_importance.tolist()

    def _build_routing_info(
        self,
        routing_events: list[dict[str, Any]],
        execution_trace: list[dict[str, Any]],
        mlp_execution_trace: list[dict[str, Any]],
        keep_mask: list[bool] | None,
        compute_units_executed: float,
        compute_units_total: float,
    ) -> dict[str, Any]:
        if routing_events:
            importance_matrix, block_importance, self_importance = self._aggregate_routing(routing_events)
        else:
            importance_matrix = torch.zeros(
                self.num_block_positions,
                self.num_block_positions,
                device=self.embeddings.weight.device,
                dtype=torch.float32,
            )
            block_importance = [0.0] * self.num_block_positions
            self_importance = [0.0] * self.num_block_positions
        num_blocks_executed = sum(entry["executed_fraction"] for entry in execution_trace)
        return {
            "importance_matrix": importance_matrix.detach(),
            "block_importance": block_importance,
            "self_importance": self_importance,
            "block_schedule": list(self.block_schedule),
            "keep_mask": keep_mask if keep_mask is not None else [True] * self.num_block_positions,
            "blocks_executed": execution_trace,
            "execution_trace": execution_trace,
            "mlp_execution_trace": mlp_execution_trace,
            "num_blocks_executed": float(num_blocks_executed),
            "effective_depth": float(num_blocks_executed),
            "expected_depth": float(num_blocks_executed),
            "num_mlp_skipped": float(sum(1 for entry in mlp_execution_trace if entry["status"] == "skipped")),
            "num_mlp_executed": float(sum(1 for entry in mlp_execution_trace if entry["status"] == "executed")),
            "num_compute_units_executed": float(compute_units_executed),
            "num_compute_units_total": float(compute_units_total),
            "compute_ratio": float(compute_units_executed / max(compute_units_total, 1.0)),
        }

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        return_routing_info: bool = False,
        enable_skipping: bool | None = None,
        skip_keep_mask: list[int] | list[bool] | torch.Tensor | None = None,
        dynamic_skip_strategy: str | None = None,
        dynamic_skip_granularity: str | None = None,
        dynamic_skip_probe_mode: str | None = None,
        dynamic_skip_threshold: float | None = None,
        dynamic_skip_position_thresholds: list[float] | torch.Tensor | None = None,
        dynamic_skip_max_skips: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | ReSkipBaseModelOutputWithPast:
        legacy_loop_kwargs = {
            "min_halt_depth": kwargs.pop("min_halt_depth", None),
            "fixed_loop_positions": kwargs.pop("fixed_loop_positions", None),
            "collect_loop_step_states": kwargs.pop("collect_loop_step_states", None),
        }
        if any(value not in (None, False, 0, 0.0) for value in legacy_loop_kwargs.values()):
            warnings.warn(
                "Ignoring legacy ReLoop-only runtime arguments on `reskip_transformer`.",
                stacklevel=2,
            )
        if output_attentions:
            warnings.warn(
                "`ReSkipTransformerModel` does not return token attention weights. "
                "`output_attentions` is forced to `False`."
            )
            output_attentions = False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if dynamic_skip_strategy is None:
            dynamic_skip_strategy = getattr(self.config, "dynamic_skip_strategy", None)
        if dynamic_skip_granularity is None:
            dynamic_skip_granularity = getattr(self.config, "dynamic_skip_granularity", "block")
        if dynamic_skip_granularity not in {"block", "mlp"}:
            raise ValueError("`dynamic_skip_granularity` must be either 'block' or 'mlp'.")
        if dynamic_skip_probe_mode is None:
            dynamic_skip_probe_mode = getattr(self.config, "dynamic_skip_probe_mode", "all")
        if dynamic_skip_threshold is None:
            dynamic_skip_threshold = getattr(self.config, "dynamic_skip_threshold", None)
        if dynamic_skip_position_thresholds is None:
            dynamic_skip_position_thresholds = self._dynamic_skip_position_thresholds
        if dynamic_skip_max_skips is None:
            dynamic_skip_max_skips = getattr(self.config, "dynamic_skip_max_skips", None)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        collect_routing_info = return_routing_info
        collect_routing_weights = collect_routing_info and not self.training
        all_hidden_states = () if output_hidden_states else None
        next_cache = past_key_values
        keep_mask = self._resolve_keep_mask(enable_skipping, skip_keep_mask)
        routing_events: list[dict[str, Any]] = []
        execution_trace: list[dict[str, Any]] = []
        mlp_execution_trace: list[dict[str, Any]] = []
        routing_entropy_terms: list[torch.Tensor] = []
        prev_block_dynamic_stats: dict[str, float | None] | None = None

        block_states: list[torch.Tensor | None] = [None] * (self.num_block_positions + 1)
        block_states[0] = inputs_embeds
        hidden_states = inputs_embeds

        dynamic_skip_position_thresholds = self._normalize_dynamic_skip_position_thresholds(
            dynamic_skip_position_thresholds,
            dynamic_skip_granularity,
        )
        dynamic_skips_taken = 0
        compute_units_total = float(self.num_block_positions * self.layers_per_block * 2)
        compute_units_executed = 0.0

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        for position, block_idx in enumerate(self.block_schedule):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            current_block = self.layers[block_idx]
            phase1_stats: dict[str, float | None] | None = None
            decision_stats: dict[str, float | None] | None = None
            phase1_outputs = None
            completed_source_ids = None
            reusable_phase1 = False
            dynamic_skip_active = False
            use_prev_block_dynamic = (
                dynamic_skip_granularity == "block"
                and is_cached_prev_dynamic_strategy(dynamic_skip_strategy)
            )
            if dynamic_skip_granularity == "block" and not use_prev_block_dynamic:
                dynamic_skip_active = self._dynamic_skip_needs_phase1(
                    dynamic_skip_strategy,
                    dynamic_skip_threshold,
                    dynamic_skip_position_thresholds,
                    position,
                    dynamic_skips_taken,
                    dynamic_skip_max_skips,
            )
            collect_dynamic_skip_stats = collect_routing_info and dynamic_skip_strategy is not None
            can_use_probe_only_stats = (
                dynamic_skip_active
                and not collect_dynamic_skip_stats
                and not collect_routing_weights
                and dynamic_skip_granularity == "block"
                and dynamic_skip_probe_mode != "all"
            )
            if can_use_probe_only_stats:
                phase1_stats = current_block.probe_phase1_stats(
                    block_states=block_states,
                    current_block_idx=position,
                    probe_mode=dynamic_skip_probe_mode,
                    strategy=dynamic_skip_strategy,
                )
                decision_stats = phase1_stats
            elif dynamic_skip_active or (collect_dynamic_skip_stats and not use_prev_block_dynamic):
                phase1_outputs, completed_source_ids = current_block.prepare_phase1(
                    block_states=block_states,
                    current_block_idx=position,
                    return_routing_weights=collect_routing_weights,
                    probe_mode=dynamic_skip_probe_mode,
                )
                phase1_stats = current_block.summarize_phase1(phase1_outputs, completed_source_ids)
                reusable_phase1 = dynamic_skip_probe_mode == "all"
                decision_stats = phase1_stats
            elif use_prev_block_dynamic:
                decision_stats = prev_block_dynamic_stats
                dynamic_skip_active = decision_stats is not None and should_skip_dynamic_position(
                    strategy=dynamic_skip_strategy,
                    threshold=dynamic_skip_threshold,
                    position_thresholds=dynamic_skip_position_thresholds,
                    phase1_stats=decision_stats,
                    position=position,
                    num_positions=self.num_block_positions,
                    skipped_so_far=dynamic_skips_taken,
                    max_skips=dynamic_skip_max_skips,
                )

            should_execute = True if keep_mask is None else keep_mask[position]
            if should_execute and dynamic_skip_active:
                if use_prev_block_dynamic:
                    should_execute = not should_skip_dynamic_position(
                        strategy=dynamic_skip_strategy,
                        threshold=dynamic_skip_threshold,
                        position_thresholds=dynamic_skip_position_thresholds,
                        phase1_stats=decision_stats if decision_stats is not None else {},
                        position=position,
                        num_positions=self.num_block_positions,
                        skipped_so_far=dynamic_skips_taken,
                        max_skips=dynamic_skip_max_skips,
                    )
                else:
                    should_execute = not self._should_skip_dynamic(
                        dynamic_skip_strategy,
                        dynamic_skip_threshold,
                        dynamic_skip_position_thresholds,
                        phase1_stats if phase1_stats is not None else {},
                        position,
                        dynamic_skips_taken,
                        dynamic_skip_max_skips,
                    )
            if not should_execute:
                dynamic_skips_taken += 1
                block_states[position + 1] = hidden_states
                if collect_routing_info:
                    payload = {
                        "position": position,
                        "block_idx": block_idx,
                        "status": "skipped",
                        "executed_fraction": 0.0,
                    }
                    if decision_stats is not None:
                        payload.update(decision_stats)
                    execution_trace.append(
                        payload
                    )
                continue

            (
                hidden_states,
                next_cache,
                attn_records,
                mlp_records,
                router_entropies,
                block_phase1_summary,
                block_layer_trace,
                block_mlp_trace,
                dynamic_skips_taken,
                block_compute_units,
            ) = current_block(
                block_states=block_states,
                current_block_idx=position,
                attention_mask=attention_mask,
                past_key_values=next_cache,
                use_cache=use_cache,
                active_mask=None,
                return_routing_weights=collect_routing_weights,
                phase1_outputs=phase1_outputs if reusable_phase1 else None,
                completed_source_ids=completed_source_ids if reusable_phase1 else None,
                cache_layer_offset=position * self.layers_per_block,
                dynamic_skip_granularity=dynamic_skip_granularity,
                dynamic_skip_strategy=dynamic_skip_strategy,
                dynamic_skip_threshold=dynamic_skip_threshold,
                dynamic_skip_position_thresholds=dynamic_skip_position_thresholds,
                dynamic_skip_max_skips=dynamic_skip_max_skips,
                dynamic_skips_taken=dynamic_skips_taken,
                collect_dynamic_skip_stats=collect_dynamic_skip_stats,
                **kwargs,
            )
            routing_entropy_terms.extend(router_entropies)
            compute_units_executed += float(block_compute_units)
            mlp_execution_trace.extend(block_mlp_trace)
            if dynamic_skip_granularity == "block" and (dynamic_skip_strategy is not None or collect_dynamic_skip_stats):
                prev_block_dynamic_stats = block_phase1_summary
            if collect_routing_weights:
                for source_ids, attn_weights in attn_records:
                    if attn_weights is not None:
                        self._record_routing(routing_events, position, "attn", source_ids, attn_weights)
                for source_ids, mlp_weights in mlp_records:
                    if mlp_weights is not None:
                        self._record_routing(routing_events, position, "mlp", source_ids, mlp_weights)
            block_states[position + 1] = hidden_states

            if collect_routing_info:
                payload = {
                    "position": position,
                    "block_idx": block_idx,
                    "status": "executed",
                    "executed_fraction": 1.0,
                }
                if decision_stats is not None:
                    payload.update(decision_stats)
                execution_trace.append(
                    payload
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        routing_info = None
        routing_entropy_tensor = None
        if routing_entropy_terms:
            routing_entropy_tensor = torch.stack(routing_entropy_terms).mean()
        if collect_routing_info:
            routing_info = self._build_routing_info(
                routing_events=routing_events,
                execution_trace=execution_trace,
                mlp_execution_trace=mlp_execution_trace,
                keep_mask=keep_mask,
                compute_units_executed=compute_units_executed,
                compute_units_total=compute_units_total,
            )
            if routing_entropy_tensor is not None:
                routing_info["routing_entropy"] = float(routing_entropy_tensor.detach().item())
        # Avoid retaining training-step autograd state on the module between
        # iterations. The structured routing payload is for metrics/analysis only.
        self._last_routing_info = None if self.training else routing_info

        if not return_dict:
            output = (
                hidden_states,
                next_cache,
                all_hidden_states,
                None,
                routing_info if return_routing_info else None,
                routing_entropy_tensor,
            )
            return tuple(item for item in output if item is not None)

        return ReSkipBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
            routing_info=routing_info if return_routing_info else None,
            routing_entropy_tensor=routing_entropy_tensor,
        )

    @torch.no_grad()
    def get_routing_statistics(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            return_routing_info=True,
            enable_skipping=False,
        )
        if outputs.routing_info is None:
            raise RuntimeError("Routing statistics were requested but not returned.")
        return outputs.routing_info


class ReSkipTransformerForCausalLM(ReSkipTransformerPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ReSkipTransformerConfig):
        super().__init__(config)
        self.model = ReSkipTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
        logits_to_keep: int | None = None,
        **kwargs,
    ):
        has_past = past_key_values is not None and len(past_key_values) > 0
        if has_past:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and not has_past:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "logits_to_keep": logits_to_keep,
            }
        )
        model_inputs.update(kwargs)
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        return_routing_info: bool = False,
        enable_skipping: bool | None = None,
        skip_keep_mask: list[int] | list[bool] | torch.Tensor | None = None,
        dynamic_skip_strategy: str | None = None,
        dynamic_skip_granularity: str | None = None,
        dynamic_skip_probe_mode: str | None = None,
        dynamic_skip_threshold: float | None = None,
        dynamic_skip_position_thresholds: list[float] | torch.Tensor | None = None,
        dynamic_skip_max_skips: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | ReSkipCausalLMOutputWithPast:
        legacy_loop_kwargs = {
            "ponder_loss_weight_override": kwargs.pop("ponder_loss_weight_override", None),
            "focused_halt_loss_weight_override": kwargs.pop("focused_halt_loss_weight_override", None),
            "min_halt_depth": kwargs.pop("min_halt_depth", None),
        }
        if any(value not in (None, False, 0, 0.0) for value in legacy_loop_kwargs.values()):
            warnings.warn(
                "Ignoring legacy ReLoop-only runtime arguments on `reskip_transformer`.",
                stacklevel=2,
            )
        if labels is not None and use_cache is None:
            use_cache = False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_routing_info=return_routing_info,
            enable_skipping=enable_skipping,
            skip_keep_mask=skip_keep_mask,
            dynamic_skip_strategy=dynamic_skip_strategy,
            dynamic_skip_granularity=dynamic_skip_granularity,
            dynamic_skip_probe_mode=dynamic_skip_probe_mode,
            dynamic_skip_threshold=dynamic_skip_threshold,
            dynamic_skip_position_thresholds=dynamic_skip_position_thresholds,
            dynamic_skip_max_skips=dynamic_skip_max_skips,
            **kwargs,
        )

        hidden_states = outputs[0]
        if self.config.fuse_linear_cross_entropy:
            logits = None
        else:
            logits_input = hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:]
            logits = self.lm_head(logits_input)

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            original_labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (original_labels[..., 1:], torch.full_like(original_labels[:, :1], criterion.ignore_index)),
                dim=1,
            )
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ReSkipCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            routing_info=outputs.routing_info if return_routing_info else None,
            routing_entropy_tensor=getattr(outputs, "routing_entropy_tensor", None),
        )
