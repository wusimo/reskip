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
    query = router.w_query.float()
    norm_weight = getattr(router.key_norm, "weight", None)
    if norm_weight is not None:
        query = query * norm_weight.float()
    return query


def _rms_norm_base(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    hidden_states_fp32 = hidden_states.float()
    inv_rms = torch.rsqrt(hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return hidden_states_fp32 * inv_rms


def batch_attend_completed_blocks(
    routers: list["BlockAttentionResidual"],
    completed_blocks: list[torch.Tensor],
    return_weights: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    values = torch.stack(completed_blocks, dim=0)
    base_keys = _rms_norm_base(values, routers[0].key_norm.eps)
    queries = torch.stack([_router_effective_query(router) for router in routers], dim=0)
    scale = math.sqrt(values.shape[-1]) * routers[0].temperature
    scores = torch.einsum("qd,nbtd->qnbt", queries, base_keys) / scale
    max_scores = scores.amax(dim=1)
    exp_scores = torch.exp(scores - max_scores.unsqueeze(1))
    lse = exp_scores.sum(dim=1)
    outputs = torch.einsum("qnbt,nbtd->qbtd", exp_scores, values.float()).to(values.dtype)

    if return_weights:
        weights = (exp_scores / lse.unsqueeze(1)).permute(0, 2, 3, 1).to(values.dtype)
    else:
        weights = None

    result = []
    for idx in range(len(routers)):
        result.append((outputs[idx], max_scores[idx], lse[idx], None if weights is None else weights[idx]))
    return result


def merge_with_partial_block(
    router: "BlockAttentionResidual",
    phase1: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    partial_block: torch.Tensor | None,
    return_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    phase1_output, phase1_max, phase1_lse, phase1_weights = phase1
    if partial_block is None:
        hidden_states = phase1_output * phase1_lse.reciprocal().to(phase1_output.dtype).unsqueeze(-1)
        return hidden_states, phase1_weights

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

    if not return_weights:
        return hidden_states, None

    if phase1_weights is None:
        raise RuntimeError("Phase-1 weights are required when return_weights=True.")
    weights = torch.cat(
        [phase1_weights * phase1_weight.unsqueeze(-1), partial_weight.unsqueeze(-1)],
        dim=-1,
    )
    return hidden_states, weights


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


@dataclass
class ReSkipBaseModelOutputWithPast(BaseModelOutputWithPast):
    routing_info: dict[str, Any] | None = None


@dataclass
class ReSkipCausalLMOutputWithPast(CausalLMOutputWithPast):
    routing_info: dict[str, Any] | None = None


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
        nn.init.zeros_(self.w_query)
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
        **kwargs: Unpack[Any],
    ) -> tuple[torch.Tensor, Cache | None]:
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, _, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
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
        **kwargs: Unpack[Any],
    ) -> tuple[
        torch.Tensor,
        Cache | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        attn_input, attn_weights = merge_with_partial_block(
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
            **kwargs,
        )
        old_partial = partial_block
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        partial_block = blend_states(
            block_input if old_partial is None else old_partial,
            partial_block,
            active_mask,
        )

        mlp_input, mlp_weights = merge_with_partial_block(
            self.mlp_router,
            mlp_phase1,
            partial_block,
            return_weights=return_routing_weights,
        )
        mlp_out = self.forward_mlp(mlp_input, **kwargs)
        old_partial = partial_block
        partial_block = partial_block + mlp_out
        partial_block = blend_states(old_partial, partial_block, active_mask)

        return partial_block, past_key_values, attn_weights, mlp_weights


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

    def forward(
        self,
        block_states: list[torch.Tensor | None],
        current_block_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        active_mask: torch.Tensor | None = None,
        return_routing_weights: bool = False,
        **kwargs: Unpack[Any],
    ) -> tuple[
        torch.Tensor,
        Cache | None,
        list[tuple[list[int], torch.Tensor | None]],
        list[tuple[list[int], torch.Tensor | None]],
    ]:
        block_input = block_states[current_block_idx]
        if block_input is None:
            raise RuntimeError("Expected the current block input to be available in block_states.")

        completed_blocks, completed_source_ids = collect_completed_blocks(block_states, current_block_idx)
        routers = []
        for layer in self.layers:
            routers.extend([layer.attn_router, layer.mlp_router])
        phase1_outputs = batch_attend_completed_blocks(
            routers=routers,
            completed_blocks=completed_blocks,
            return_weights=return_routing_weights,
        )

        partial_block = None
        next_cache = past_key_values
        attn_records: list[tuple[list[int], torch.Tensor | None]] = []
        mlp_records: list[tuple[list[int], torch.Tensor | None]] = []

        for layer_idx, layer in enumerate(self.layers):
            attn_phase1 = phase1_outputs[2 * layer_idx]
            mlp_phase1 = phase1_outputs[2 * layer_idx + 1]
            attn_source_ids = list(completed_source_ids) if partial_block is None else [*completed_source_ids, current_block_idx]
            partial_block, next_cache, attn_weights, mlp_weights = layer(
                block_input=block_input,
                partial_block=partial_block,
                attn_phase1=attn_phase1,
                mlp_phase1=mlp_phase1,
                attention_mask=attention_mask,
                past_key_values=next_cache,
                use_cache=use_cache,
                active_mask=active_mask,
                return_routing_weights=return_routing_weights,
                **kwargs,
            )
            attn_records.append((attn_source_ids, attn_weights))
            mlp_records.append(([*completed_source_ids, current_block_idx], mlp_weights))

        return partial_block, next_cache, attn_records, mlp_records


class ReSkipTransformerPreTrainedModel(PreTrainedModel):
    config_class = ReSkipTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReSkipBlockGroup", "ReSkipTransformerLayer"]
    _supports_cache_class = True

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
        self.num_unique_blocks = (
            config.num_recurrent_blocks if config.enable_looping else config.attn_res_num_blocks
        )
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
        self.halt_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.halt_head = nn.Linear(config.hidden_size, 1, bias=True)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self._skip_keep_mask = self._normalize_keep_mask(config.skip_keep_mask)
        self._last_routing_info: dict[str, Any] | None = None

        self.post_init()

    def _build_block_schedule(self) -> list[int]:
        if not self.config.enable_looping:
            return list(range(self.config.attn_res_num_blocks))
        return [position % self.config.num_recurrent_blocks for position in range(self.config.attn_res_num_blocks)]

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

    def _pool_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

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
        keep_mask: list[bool] | None,
        halt_probabilities: list[float] | None,
        ponder_cost: float,
    ) -> dict[str, Any]:
        importance_matrix, block_importance, self_importance = self._aggregate_routing(routing_events)
        num_blocks_executed = sum(entry["executed_fraction"] for entry in execution_trace)
        return {
            "importance_matrix": importance_matrix,
            "block_importance": block_importance,
            "self_importance": self_importance,
            "block_schedule": list(self.block_schedule),
            "keep_mask": keep_mask if keep_mask is not None else [True] * self.num_block_positions,
            "blocks_executed": execution_trace,
            "execution_trace": execution_trace,
            "num_blocks_executed": float(num_blocks_executed),
            "effective_depth": float(num_blocks_executed),
            "halt_probabilities": halt_probabilities,
            "ponder_cost": float(ponder_cost),
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
        **kwargs: Unpack[Any],
    ) -> tuple | ReSkipBaseModelOutputWithPast:
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

        if self.config.enable_looping and use_cache:
            logger.warning_once("Looping mode disables KV cache because shared blocks reuse logical layer indices.")
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        collect_routing_info = return_routing_info
        all_hidden_states = () if output_hidden_states else None
        next_cache = past_key_values
        keep_mask = self._resolve_keep_mask(enable_skipping, skip_keep_mask)
        routing_events: list[dict[str, Any]] = []
        execution_trace: list[dict[str, Any]] = []

        block_states: list[torch.Tensor | None] = [None] * (self.num_block_positions + 1)
        block_states[0] = inputs_embeds
        hidden_states = inputs_embeds

        batch_size = hidden_states.shape[0]
        halt_cumulative = hidden_states.new_zeros(batch_size)
        halt_probabilities: list[float] | None = [] if self.config.enable_looping else None
        ponder_cost = 0.0

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        for position, block_idx in enumerate(self.block_schedule):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            active_mask = None
            executed_fraction = 0.0
            if self.config.enable_looping:
                active_mask = halt_cumulative < self.config.halt_threshold
                executed_fraction = float(active_mask.float().mean().item())
                if not torch.any(active_mask):
                    if collect_routing_info:
                        execution_trace.append(
                            {
                                "position": position,
                                "block_idx": block_idx,
                                "status": "halted",
                                "executed_fraction": 0.0,
                                "halt_probability": 0.0,
                            }
                        )
                    continue
            else:
                executed_fraction = 1.0

            should_execute = True if keep_mask is None else keep_mask[position]
            if not should_execute:
                if collect_routing_info:
                    execution_trace.append(
                        {
                            "position": position,
                            "block_idx": block_idx,
                            "status": "skipped",
                            "executed_fraction": 0.0,
                            "halt_probability": 0.0,
                        }
                    )
                continue

            current_block = self.layers[block_idx]
            hidden_states, next_cache, attn_records, mlp_records = current_block(
                block_states=block_states,
                current_block_idx=position,
                attention_mask=attention_mask,
                past_key_values=next_cache,
                use_cache=use_cache,
                active_mask=active_mask,
                return_routing_weights=collect_routing_info,
                **kwargs,
            )
            if collect_routing_info:
                for source_ids, attn_weights in attn_records:
                    self._record_routing(routing_events, position, "attn", source_ids, attn_weights)
                for source_ids, mlp_weights in mlp_records:
                    self._record_routing(routing_events, position, "mlp", source_ids, mlp_weights)
            block_states[position + 1] = hidden_states

            halt_probability_mean = 0.0
            if self.config.enable_looping:
                pooled = self._pool_hidden(self.halt_norm(hidden_states), attention_mask)
                block_halt = torch.sigmoid(self.halt_head(pooled)).squeeze(-1)
                if active_mask is not None:
                    block_halt = block_halt * active_mask.to(block_halt.dtype)
                    halt_cumulative = torch.clamp(halt_cumulative + block_halt, max=1.0)
                else:
                    halt_cumulative = torch.clamp(halt_cumulative + block_halt, max=1.0)
                halt_probability_mean = float(block_halt.mean().item())
                halt_probabilities.append(halt_probability_mean)
                ponder_cost += executed_fraction

            if collect_routing_info:
                execution_trace.append(
                    {
                        "position": position,
                        "block_idx": block_idx,
                        "status": "executed",
                        "executed_fraction": executed_fraction,
                        "halt_probability": halt_probability_mean,
                    }
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        routing_info = None
        if collect_routing_info:
            routing_info = self._build_routing_info(
                routing_events=routing_events,
                execution_trace=execution_trace,
                keep_mask=keep_mask,
                halt_probabilities=halt_probabilities,
                ponder_cost=ponder_cost,
            )
        self._last_routing_info = routing_info

        if not return_dict:
            output = (hidden_states, next_cache, all_hidden_states, None, routing_info if return_routing_info else None)
            return tuple(item for item in output if item is not None)

        return ReSkipBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
            routing_info=routing_info if return_routing_info else None,
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
        **kwargs: Unpack[Any],
    ) -> tuple | ReSkipCausalLMOutputWithPast:
        if labels is not None and use_cache is None:
            use_cache = False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        need_routing_info = return_routing_info or (
            labels is not None and self.config.enable_looping and self.config.ponder_loss_weight > 0
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_routing_info=need_routing_info,
            enable_skipping=enable_skipping,
            skip_keep_mask=skip_keep_mask,
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
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)),
                dim=1,
            )
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

            routing_info = getattr(outputs, "routing_info", None)
            if routing_info is not None and self.config.enable_looping and self.config.ponder_loss_weight > 0:
                loss = loss + self.config.ponder_loss_weight * hidden_states.new_tensor(routing_info["ponder_cost"])

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
        )
