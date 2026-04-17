from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.models.reloop_transformer.configuration_reloop_transformer import ReLoopTransformerConfig
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


def _normalize_loop_position_limit(limit: int | None, max_positions: int) -> int | None:
    if limit is None:
        return None
    return max(1, min(int(limit), max_positions))


def blend_states(
    old_states: torch.Tensor,
    new_states: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    if active_mask is None:
        return new_states
    # active_mask can be either per-sequence [batch] or per-token [batch, seq]
    if active_mask.dim() == 1:
        mask = active_mask[:, None, None].to(dtype=new_states.dtype)
    elif active_mask.dim() == 2:
        mask = active_mask.unsqueeze(-1).to(dtype=new_states.dtype)
    else:
        raise ValueError(f"active_mask has unexpected ndim={active_mask.dim()}")
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


def per_sample_recent_weight(
    routers: list["BlockAttentionResidual"],
    completed_blocks: list[torch.Tensor],
    per_token: bool = False,
    detach: bool = True,
) -> torch.Tensor:
    """Routing weight on the most recent completed block, averaged over routers.

    Returns [batch] (per-sequence, averaged over seq) or [batch, seq] (per-token).
    Low value → the recent block is not contributing much → good candidate for halting.
    Pass detach=False to get a differentiable version for ponder cost computation.
    """
    if len(completed_blocks) <= 1:
        base = completed_blocks[0]
        if per_token:
            return base.new_zeros(base.shape[0], base.shape[1])
        return base.new_zeros(base.shape[0])
    values = torch.stack(completed_blocks, dim=0)  # [sources, batch, seq, hidden]
    base_keys = _rms_norm_base(values, routers[0].key_norm.eps)
    queries = torch.stack([_router_effective_query(r) for r in routers], dim=0)
    scale = math.sqrt(values.shape[-1]) * routers[0].temperature
    scores = torch.einsum("qd,nbtd->qnbt", queries, base_keys) / scale
    probs = torch.softmax(scores, dim=1)  # [routers, sources, batch, seq]
    recent_probs = probs[:, -1, :, :]  # [routers, batch, seq]
    if per_token:
        result = recent_probs.mean(dim=0)  # [batch, seq]
    else:
        result = recent_probs.mean(dim=(0, 2))  # [batch]
    return result.detach() if detach else result


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


def phase1_feature_tensor(
    phase1_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor],
    source_ids: list[int],
    *,
    detach: bool,
) -> torch.Tensor:
    _output, _max, _lse, entropy, _weights, source_means = phase1_output
    mean_entropy = normalize_router_entropy(entropy, len(source_ids)).mean().float()
    mean_embed = source_means[0].float() if source_means.numel() > 0 else mean_entropy.new_zeros(())
    mean_recent = source_means[-1].float() if len(source_ids) > 1 else mean_entropy.new_zeros(())
    features = torch.stack(
        [
            mean_recent,
            mean_embed,
            1.0 - mean_entropy,
        ]
    )
    if detach:
        features = features.detach()
    return features


@dataclass
class ReLoopBaseModelOutputWithPast(BaseModelOutputWithPast):
    routing_info: dict[str, Any] | None = None
    ponder_cost_tensor: torch.Tensor | None = None
    expected_depth_tensor: torch.Tensor | None = None
    exit_kl_tensor: torch.Tensor | None = None
    exit_entropy_tensor: torch.Tensor | None = None
    early_exit_mass_tensor: torch.Tensor | None = None
    routing_entropy_tensor: torch.Tensor | None = None
    loop_step_hidden_states: tuple[torch.Tensor, ...] | None = None
    halt_logits_tensors: tuple[torch.Tensor, ...] | None = None
    multi_exit_hidden_states: tuple[torch.Tensor, ...] | None = None
    ponder_expected_depth_tensor: torch.Tensor | None = None


@dataclass
class ReLoopCausalLMOutputWithPast(CausalLMOutputWithPast):
    routing_info: dict[str, Any] | None = None
    ponder_cost_tensor: torch.Tensor | None = None
    expected_depth_tensor: torch.Tensor | None = None
    exit_kl_tensor: torch.Tensor | None = None
    exit_entropy_tensor: torch.Tensor | None = None
    early_exit_mass_tensor: torch.Tensor | None = None
    routing_entropy_tensor: torch.Tensor | None = None
    focused_halt_loss_tensor: torch.Tensor | None = None
    focused_halt_target_mean_tensor: torch.Tensor | None = None
    focused_halt_improvement_mean_tensor: torch.Tensor | None = None


class BlockAttentionResidual(nn.Module):
    """Block AttnRes over completed blocks plus the current partial block."""

    def __init__(self, config: ReLoopTransformerConfig):
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

class ReLoopTransformerLayer(GradientCheckpointingLayer):
    """Standard transformer layer split into attention and MLP phases."""

    def __init__(self, config: ReLoopTransformerConfig, layer_idx: int):
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
        cache_layer_idx: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple[
        torch.Tensor,
        Cache | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
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

        return partial_block, past_key_values, attn_weights, mlp_weights, attn_entropy, mlp_entropy


class ReLoopBlockGroup(nn.Module):
    def __init__(
        self,
        config: ReLoopTransformerConfig,
        block_idx: int,
        layers_per_block: int,
        first_layer_idx: int,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.layers = nn.ModuleList(
            [ReLoopTransformerLayer(config, first_layer_idx + offset) for offset in range(layers_per_block)]
        )

    def prepare_phase1(
        self,
        block_states: list[torch.Tensor | None],
        current_block_idx: int,
        return_routing_weights: bool = False,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]],
        list[int],
    ]:
        completed_blocks, completed_source_ids = collect_completed_blocks(block_states, current_block_idx)
        routers = []
        for layer in self.layers:
            routers.extend([layer.attn_router, layer.mlp_router])
        phase1_outputs = batch_attend_completed_blocks(
            routers=routers,
            completed_blocks=completed_blocks,
            return_weights=return_routing_weights,
        )
        return phase1_outputs, completed_source_ids

    def build_halt_features(
        self,
        phase1_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]],
        completed_source_ids: list[int],
        *,
        position: int,
        num_positions: int,
        detach: bool,
    ) -> torch.Tensor:
        if detach:
            feature_rows: list[tuple[float, float, float]] = []
            for item in phase1_outputs:
                _output, _max, _lse, entropy, _weights, source_means = item
                mean_entropy = float(normalize_router_entropy(entropy, len(completed_source_ids)).mean().detach().item())
                mean_embed = float(source_means[0].detach().float().item()) if source_means.numel() > 0 else 0.0
                mean_recent = float(source_means[-1].detach().float().item()) if len(completed_source_ids) > 1 else 0.0
                feature_rows.append((mean_recent, mean_embed, 1.0 - mean_entropy))
            if not feature_rows:
                raise ValueError("Expected at least one phase-1 output when building halt features.")
            feature_mean = [
                sum(row[idx] for row in feature_rows) / len(feature_rows)
                for idx in range(3)
            ]
            progress = float(position) / max(float(num_positions - 1), 1.0)
            source_fraction = float(len(completed_source_ids)) / max(float(num_positions), 1.0)
            return phase1_outputs[0][0].new_tensor([*feature_mean, source_fraction, progress])

        per_router = [
            phase1_feature_tensor(item, completed_source_ids, detach=detach)
            for item in phase1_outputs
        ]
        feature_mean = torch.stack(per_router, dim=0).mean(dim=0)
        progress = feature_mean.new_tensor(float(position) / max(float(num_positions - 1), 1.0))
        source_fraction = feature_mean.new_tensor(float(len(completed_source_ids)) / max(float(num_positions), 1.0))
        return torch.cat([feature_mean, torch.stack([source_fraction, progress])], dim=0)

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
        **kwargs: Unpack[Any],
    ) -> tuple[
        torch.Tensor,
        Cache | None,
        list[tuple[list[int], torch.Tensor | None]],
        list[tuple[list[int], torch.Tensor | None]],
        list[torch.Tensor],
        list[dict[str, Any]],
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
        need_router_entropy = self.training
        router_entropies: list[torch.Tensor] = []
        mlp_execution_trace: list[dict[str, Any]] = []
        compute_units_executed = 0

        for layer_idx, layer in enumerate(self.layers):
            attn_phase1 = phase1_outputs[2 * layer_idx]
            mlp_phase1 = phase1_outputs[2 * layer_idx + 1]
            attn_source_ids = list(completed_source_ids) if partial_block is None else [*completed_source_ids, current_block_idx]
            cache_layer_idx = None if cache_layer_offset is None else cache_layer_offset + layer_idx
            partial_block, next_cache, attn_weights, mlp_weights, attn_entropy, mlp_entropy = layer(
                block_input=block_input,
                partial_block=partial_block,
                attn_phase1=attn_phase1,
                mlp_phase1=mlp_phase1,
                attention_mask=attention_mask,
                past_key_values=next_cache,
                use_cache=use_cache,
                active_mask=active_mask,
                return_routing_weights=return_routing_weights,
                cache_layer_idx=cache_layer_idx,
                **kwargs,
            )
            attn_records.append((attn_source_ids, attn_weights))
            mlp_records.append(([*completed_source_ids, current_block_idx], mlp_weights))
            compute_units_executed += 2
            attn_num_sources = len(attn_source_ids)
            mlp_num_sources = len(completed_source_ids) + 1
            if need_router_entropy:
                router_entropies.extend(
                    [
                        normalize_router_entropy(attn_entropy, attn_num_sources).mean(),
                        normalize_router_entropy(mlp_entropy, mlp_num_sources).mean(),
                    ]
                )
            mlp_execution_trace.append(
                {
                    "position": layer.layer_idx,
                    "block_position": current_block_idx,
                    "block_idx": self.block_idx,
                    "local_layer_idx": layer_idx,
                    "status": "executed",
                    "executed_fraction": 1.0,
                }
            )

        return (
            partial_block,
            next_cache,
            attn_records,
            mlp_records,
            router_entropies,
            mlp_execution_trace,
            compute_units_executed,
        )


class ReLoopTransformerPreTrainedModel(PreTrainedModel):
    config_class = ReLoopTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReLoopBlockGroup", "ReLoopTransformerLayer"]
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
            if getattr(module, "_is_loop_halt_head", False):
                target_bias = self._compute_initial_halt_bias()
                nn.init.constant_(module.bias, target_bias)
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

    def _compute_initial_halt_bias(self) -> float:
        target_prob = self.config.halt_threshold / (self.config.attn_res_num_blocks + 1)
        target_prob = min(max(target_prob, 1e-4), 1.0 - 1e-4)
        return math.log(target_prob / (1.0 - target_prob))


class ReLoopTransformerModel(ReLoopTransformerPreTrainedModel):
    def __init__(self, config: ReLoopTransformerConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers_per_block = config.num_hidden_layers // config.attn_res_num_blocks
        self.num_block_positions = config.attn_res_num_blocks
        self.num_unique_blocks = config.num_recurrent_blocks
        self.block_schedule = self._build_block_schedule()

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                ReLoopBlockGroup(
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
        self.halt_head._is_loop_halt_head = True
        self.halt_phase1_proj = nn.Linear(5, 1, bias=False)
        self.halt_position_bias = nn.Parameter(torch.zeros(self.num_block_positions))
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False

        self.post_init()
        self._initialize_halt_head()

    def _initialize_halt_head(self) -> None:
        # Start close to full depth so halting learns to shorten computation,
        # instead of collapsing to shallow execution from random initialization.
        target_bias = self._compute_initial_halt_bias()
        nn.init.constant_(self.halt_head.bias, target_bias)
        nn.init.zeros_(self.halt_phase1_proj.weight)
        nn.init.zeros_(self.halt_position_bias)

    def _build_block_schedule(self) -> list[int]:
        return [position % self.config.num_recurrent_blocks for position in range(self.config.attn_res_num_blocks)]

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
        mlp_execution_trace: list[dict[str, Any]],
        halt_probabilities: list[float] | None,
        ponder_cost: torch.Tensor,
        expected_depth: torch.Tensor | None,
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
            "blocks_executed": execution_trace,
            "execution_trace": execution_trace,
            "mlp_execution_trace": mlp_execution_trace,
            "num_blocks_executed": float(num_blocks_executed),
            "effective_depth": float(num_blocks_executed),
            "expected_depth": (
                float(expected_depth.detach().item()) if expected_depth is not None else float(num_blocks_executed)
            ),
            "num_compute_units_executed": float(compute_units_executed),
            "num_compute_units_total": float(compute_units_total),
            "compute_ratio": float(compute_units_executed / max(compute_units_total, 1.0)),
            "halt_probabilities": halt_probabilities,
            "ponder_cost": float(ponder_cost.detach().item()),
        }

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _resolve_loop_cache_positions(
        self,
        past_key_values: Cache | None,
        explicit_limit: int | None,
    ) -> int | None:
        normalized_limit = _normalize_loop_position_limit(explicit_limit, self.num_block_positions)
        if normalized_limit is not None:
            return normalized_limit
        if past_key_values is None:
            return None
        cached_limit = getattr(past_key_values, "_loop_decode_positions", None)
        return _normalize_loop_position_limit(cached_limit, self.num_block_positions)

    def _update_loop_cache_metadata(
        self,
        past_key_values: Cache | None,
        loop_positions: int | None,
    ) -> None:
        if past_key_values is None or loop_positions is None:
            return
        past_key_values._loop_decode_positions = int(loop_positions)
        past_key_values._loop_cache_policy = "sequence_fixed_depth"

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
        min_halt_depth: int | None = None,
        fixed_loop_positions: int | None = None,
        collect_loop_step_states: bool = False,
        collect_multi_exit_states: bool = False,
        **kwargs: Unpack[Any],
    ) -> tuple | ReLoopBaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`ReLoopTransformerModel` does not return token attention weights. "
                "`output_attentions` is forced to `False`."
            )
            output_attentions = False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        collect_routing_weights = return_routing_info and not self.training
        all_hidden_states = () if output_hidden_states else None
        loop_step_hidden_states: list[torch.Tensor] | None = [] if collect_loop_step_states else None
        halt_logits_tensors: list[torch.Tensor] | None = [] if collect_loop_step_states else None
        multi_exit_hidden_states: list[torch.Tensor] | None = [] if collect_multi_exit_states else None
        next_cache = past_key_values
        routing_events: list[dict[str, Any]] = []
        execution_trace: list[dict[str, Any]] = []
        mlp_execution_trace: list[dict[str, Any]] = []
        routing_entropy_terms: list[torch.Tensor] = []

        block_states: list[torch.Tensor | None] = [None] * (self.num_block_positions + 1)
        block_states[0] = inputs_embeds
        hidden_states = inputs_embeds

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        per_token_halt = bool(self.config.attnres_halt_per_token and self.config.halt_mode == "attnres")
        if use_cache and batch_size > 1:
            raise ValueError("Looping KV cache currently supports batch size 1 only.")
        if per_token_halt:
            halt_cumulative = hidden_states.new_zeros(batch_size, seq_len)
        else:
            halt_cumulative = hidden_states.new_zeros(batch_size)
        soft_threshold = float(self.config.halt_threshold)
        # When training_full_depth is on, all blocks always execute so the
        # soft-halting machinery (which produces gradients through halted
        # blocks for ACT-style training) is not needed and can cause shape
        # conflicts with per-token halt.
        _use_soft_remaining = (
            self.training and not getattr(self.config, "training_full_depth", False)
        )
        soft_remaining = (
            hidden_states.new_full((batch_size,), soft_threshold) if _use_soft_remaining else None
        )
        # Accumulate per-token survival probability for ponder cost computation
        ponder_cost_weight_cfg = float(getattr(self.config, "ponder_cost_weight", 0.0))
        need_ponder_cost_tracking = bool(ponder_cost_weight_cfg > 0 and self.training and per_token_halt)
        ponder_expected_depth_per_token = (
            hidden_states.new_zeros(batch_size, seq_len) if need_ponder_cost_tracking else None
        )
        halt_probabilities: list[float] | None = [] if return_routing_info and not self.training else None
        ponder_cost = hidden_states.new_zeros(())
        expected_depth = hidden_states.new_zeros(())
        exit_distribution_terms: list[torch.Tensor] | None = [] if self.training else None
        early_exit_mass = hidden_states.new_zeros(()) if self.training else None
        min_halt_depth = None if min_halt_depth is None else max(1, min(int(min_halt_depth), self.num_block_positions))
        loop_cache_positions = self._resolve_loop_cache_positions(past_key_values, fixed_loop_positions)
        compute_units_total = float(self.num_block_positions * self.layers_per_block * 2)
        compute_units_executed = 0.0

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        for position, block_idx in enumerate(self.block_schedule):
            if loop_cache_positions is not None and position >= loop_cache_positions:
                break
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            active_mask = None
            active_gate = None
            executed_fraction = 0.0
            expected_fraction = None
            soft_fraction_value = None
            use_soft_min_halt = self.config.training_soft_min_halt and self.training
            hard_halt_allowed = min_halt_depth is None or (position + 1) >= min_halt_depth
            soft_halt_allowed = hard_halt_allowed or use_soft_min_halt
            if hard_halt_allowed:
                active_mask = halt_cumulative < self.config.halt_threshold
                executed_fraction = float(active_mask.float().mean().item())
                if self.training:
                    if soft_remaining is None:
                        # training_full_depth mode: soft halting is disabled.
                        # Use constant expected_fraction = 1 (all blocks always
                        # execute); the halt behavior at inference is driven by
                        # AttnRes routing independently.
                        active_gate = active_mask.to(hidden_states.dtype)
                        soft_fraction_value = hidden_states.new_ones(())
                        expected_fraction = soft_fraction_value
                    elif soft_halt_allowed:
                        active_soft = (soft_remaining / max(soft_threshold, 1e-6)).clamp_(0.0, 1.0)
                        active_gate = active_soft + (active_mask.to(hidden_states.dtype) - active_soft).detach()
                        soft_fraction_value = active_soft.mean()
                        expected_fraction = (
                            executed_fraction + soft_fraction_value - soft_fraction_value.detach()
                        )
                    else:
                        active_gate = active_mask.to(hidden_states.dtype)
                        soft_fraction_value = active_gate.mean()
                        expected_fraction = soft_fraction_value
                else:
                    active_gate = active_mask.to(hidden_states.dtype)
                    soft_fraction_value = active_gate.mean()
                    expected_fraction = soft_fraction_value
            else:
                active_mask = torch.ones_like(halt_cumulative, dtype=torch.bool)
                executed_fraction = 1.0
                active_gate = hidden_states.new_ones(batch_size)
                soft_fraction_value = active_gate.mean()
                expected_fraction = soft_fraction_value
            # During training with training_full_depth, force all blocks to
            # execute so every block receives gradient through the final LM
            # loss.  Halt probabilities are still computed (used for ponder
            # cost and inference-time actual halting).
            if self.training and self.config.training_full_depth:
                active_mask = torch.ones_like(halt_cumulative, dtype=torch.bool)
                executed_fraction = 1.0
                if per_token_halt:
                    active_gate = hidden_states.new_ones(batch_size, seq_len)
                else:
                    active_gate = hidden_states.new_ones(batch_size)

            if not torch.any(active_mask) and not self.training:
                if return_routing_info:
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

            current_block = self.layers[block_idx]
            phase1_outputs = None
            completed_source_ids = None
            halt_phase1_features = None

            if self.config.halt_use_phase1_stats:
                phase1_outputs, completed_source_ids = current_block.prepare_phase1(
                    block_states=block_states,
                    current_block_idx=position,
                    return_routing_weights=collect_routing_weights,
                )
                halt_phase1_features = current_block.build_halt_features(
                    phase1_outputs=phase1_outputs,
                    completed_source_ids=completed_source_ids,
                    position=position,
                    num_positions=self.num_block_positions,
                    detach=self.config.halt_detach_phase1_stats,
                )

            (
                hidden_states,
                next_cache,
                attn_records,
                mlp_records,
                router_entropies,
                block_mlp_trace,
                block_compute_units,
            ) = current_block(
                block_states=block_states,
                current_block_idx=position,
                attention_mask=attention_mask,
                past_key_values=next_cache,
                use_cache=use_cache,
                active_mask=active_gate,
                return_routing_weights=collect_routing_weights,
                phase1_outputs=phase1_outputs,
                completed_source_ids=completed_source_ids,
                cache_layer_offset=position * self.layers_per_block,
                **kwargs,
            )
            routing_entropy_terms.extend(router_entropies)
            compute_units_executed += float(block_compute_units)
            mlp_execution_trace.extend(block_mlp_trace)
            if collect_routing_weights:
                for source_ids, attn_weights in attn_records:
                    if attn_weights is not None:
                        self._record_routing(routing_events, position, "attn", source_ids, attn_weights)
                for source_ids, mlp_weights in mlp_records:
                    if mlp_weights is not None:
                        self._record_routing(routing_events, position, "mlp", source_ids, mlp_weights)
            block_states[position + 1] = hidden_states

            halt_probability_mean = 0.0
            halt_structural_bias = 0.0
            halt_position_bias = 0.0
            if self.config.halt_mode == "attnres":
                # ── AttnRes-native halting ──
                # Use the per-sample routing weight on the most recent
                # completed block as the halt signal.  Low weight → recent
                # block not contributing → halt.  No learned halt head;
                # the signal comes purely from AttnRes routing.
                completed_blocks_for_halt, _ = collect_completed_blocks(block_states, position)
                routers_for_halt = []
                for layer in current_block.layers:
                    routers_for_halt.extend([layer.attn_router, layer.mlp_router])
                if position > 0 and len(completed_blocks_for_halt) > 1:
                    recent_w = per_sample_recent_weight(
                        routers_for_halt, completed_blocks_for_halt, per_token=per_token_halt,
                    )
                    halt_logits = (
                        (self.config.attnres_halt_threshold - recent_w)
                        / max(self.config.attnres_halt_temperature, 1e-6)
                    )
                    # For ponder cost, compute a differentiable halt_prob that
                    # lets gradient flow back to the routing pseudo-queries.
                    # Subsample tokens to avoid ~1 GB/position of fp32 copies
                    # of the full [num_sources, batch, seq, hidden] values.
                    if need_ponder_cost_tracking:
                        ponder_n_tokens = 256
                        if seq_len > ponder_n_tokens:
                            sample_idx = torch.randperm(
                                seq_len, device=hidden_states.device,
                            )[:ponder_n_tokens]
                            sliced_blocks = [b[:, sample_idx, :] for b in completed_blocks_for_halt]
                        else:
                            sample_idx = None
                            sliced_blocks = completed_blocks_for_halt
                        recent_w_grad = per_sample_recent_weight(
                            routers_for_halt, sliced_blocks,
                            per_token=True, detach=False,
                        )
                        halt_logits_grad = (
                            (self.config.attnres_halt_threshold - recent_w_grad)
                            / max(self.config.attnres_halt_temperature, 1e-6)
                        )
                        halt_prob_grad = torch.sigmoid(halt_logits_grad).to(
                            ponder_expected_depth_per_token.dtype
                        )
                        # Expected survival past this position; accumulate on
                        # subsampled tokens only (final loss is mean).
                        if sample_idx is None:
                            ponder_expected_depth_per_token = (
                                ponder_expected_depth_per_token + (1.0 - halt_prob_grad)
                            )
                        else:
                            ponder_expected_depth_per_token = ponder_expected_depth_per_token.clone()
                            ponder_expected_depth_per_token[:, sample_idx] = (
                                ponder_expected_depth_per_token[:, sample_idx] + (1.0 - halt_prob_grad)
                            )
                else:
                    if per_token_halt:
                        halt_logits = hidden_states.new_full((batch_size, seq_len), -10.0)
                    else:
                        halt_logits = hidden_states.new_full((batch_size,), -10.0)
                block_halt = torch.sigmoid(halt_logits)
            else:
                # ── Learned halt head (original) ──
                pooled = self._pool_hidden(self.halt_norm(hidden_states), attention_mask)
                halt_logits = self.halt_head(pooled).squeeze(-1)
                if self.config.halt_use_position_bias:
                    position_bias_value = self.halt_position_bias[position].to(halt_logits.dtype)
                    halt_logits = halt_logits + position_bias_value
                    halt_position_bias = float(position_bias_value.detach().item())
                if halt_phase1_features is not None:
                    phase1_features = halt_phase1_features.to(device=pooled.device, dtype=pooled.dtype)
                    structural_logits = self.halt_phase1_proj(
                        phase1_features.unsqueeze(0).expand(pooled.shape[0], -1)
                    ).squeeze(-1)
                    halt_logits = halt_logits + structural_logits
                    halt_structural_bias = float(structural_logits.mean().detach().item())
                block_halt = torch.sigmoid(halt_logits)
            hard_block_halt = block_halt * active_mask.to(block_halt.dtype)
            halt_cumulative = torch.clamp(halt_cumulative + hard_block_halt, max=1.0)
            if soft_remaining is not None:
                if position == self.num_block_positions - 1:
                    soft_halt = soft_remaining
                    soft_remaining = torch.zeros_like(soft_remaining)
                else:
                    soft_halt = torch.minimum(soft_remaining, block_halt)
                    soft_remaining = torch.clamp(soft_remaining - soft_halt, min=0.0)
                if exit_distribution_terms is not None:
                    exit_distribution_terms.append((soft_halt / max(soft_threshold, 1e-6)).mean())
                if (
                    early_exit_mass is not None
                    and min_halt_depth is not None
                    and (position + 1) < min_halt_depth
                ):
                    early_exit_mass = early_exit_mass + (soft_halt / max(soft_threshold, 1e-6)).mean()
            if halt_probabilities is not None:
                halt_probability_mean = float(hard_block_halt.float().mean().detach().cpu())
                halt_probabilities.append(halt_probability_mean)
            if expected_fraction is None or soft_fraction_value is None:
                raise RuntimeError("Expected a differentiable execution fraction in looping mode.")
            ponder_cost = ponder_cost + expected_fraction
            expected_depth = expected_depth + soft_fraction_value
            if loop_step_hidden_states is not None:
                # Focused halt supervision only uses these states as
                # detached teacher targets. Keeping the live tensors here
                # defeats activation checkpointing and inflates memory.
                loop_step_hidden_states.append(hidden_states.detach())
            if halt_logits_tensors is not None:
                halt_logits_tensors.append(halt_logits)
            if multi_exit_hidden_states is not None:
                multi_exit_hidden_states.append(hidden_states)

            if return_routing_info:
                payload = {
                    "position": position,
                    "block_idx": block_idx,
                    "status": "executed",
                    "executed_fraction": executed_fraction,
                    "halt_probability": halt_probability_mean,
                    "halt_structural_bias": halt_structural_bias,
                    "halt_position_bias": halt_position_bias,
                }
                execution_trace.append(
                    payload
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        routing_info = None
        exit_kl_tensor = None
        exit_entropy_tensor = None
        routing_entropy_tensor = None
        if routing_entropy_terms:
            routing_entropy_tensor = torch.stack(routing_entropy_terms).mean()
        if exit_distribution_terms:
            exit_distribution = torch.stack(exit_distribution_terms)
            exit_distribution = exit_distribution / exit_distribution.sum().clamp_min(1e-8)
            exit_log_probs = torch.log(exit_distribution.clamp_min(1e-8))
            exit_entropy_tensor = -(exit_distribution * exit_log_probs).sum()
            uniform_log_prob = -math.log(exit_distribution.numel())
            exit_kl_tensor = (exit_distribution * (exit_log_probs - uniform_log_prob)).sum()
        if return_routing_info:
            routing_info = self._build_routing_info(
                routing_events=routing_events,
                execution_trace=execution_trace,
                mlp_execution_trace=mlp_execution_trace,
                halt_probabilities=halt_probabilities,
                ponder_cost=ponder_cost,
                expected_depth=expected_depth,
                compute_units_executed=compute_units_executed,
                compute_units_total=compute_units_total,
            )
            if routing_entropy_tensor is not None:
                routing_info["routing_entropy"] = float(routing_entropy_tensor.detach().item())
        if use_cache:
            cached_depth = loop_cache_positions
            if cached_depth is None:
                if routing_info is not None:
                    cached_depth = max(1, min(int(math.ceil(routing_info["effective_depth"])), self.num_block_positions))
                else:
                    cached_depth = max(1, min(len(execution_trace), self.num_block_positions))
            self._update_loop_cache_metadata(next_cache, cached_depth)

        if not return_dict:
            output = (
                hidden_states,
                next_cache,
                all_hidden_states,
                None,
                routing_info if return_routing_info else None,
                ponder_cost,
                expected_depth,
                routing_entropy_tensor,
                tuple(loop_step_hidden_states) if loop_step_hidden_states is not None else None,
                tuple(halt_logits_tensors) if halt_logits_tensors is not None else None,
            )
            return tuple(item for item in output if item is not None)

        return ReLoopBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
            routing_info=routing_info if return_routing_info else None,
            ponder_cost_tensor=ponder_cost,
            expected_depth_tensor=expected_depth,
            exit_kl_tensor=exit_kl_tensor,
            exit_entropy_tensor=exit_entropy_tensor,
            early_exit_mass_tensor=early_exit_mass,
            routing_entropy_tensor=routing_entropy_tensor,
            loop_step_hidden_states=tuple(loop_step_hidden_states) if loop_step_hidden_states is not None else None,
            halt_logits_tensors=tuple(halt_logits_tensors) if halt_logits_tensors is not None else None,
            multi_exit_hidden_states=tuple(multi_exit_hidden_states) if multi_exit_hidden_states is not None else None,
            ponder_expected_depth_tensor=(
                ponder_expected_depth_per_token.mean() if ponder_expected_depth_per_token is not None else None
            ),
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
        )
        if outputs.routing_info is None:
            raise RuntimeError("Routing statistics were requested but not returned.")
        return outputs.routing_info


class ReLoopTransformerForCausalLM(ReLoopTransformerPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ReLoopTransformerConfig):
        super().__init__(config)
        self.model = ReLoopTransformerModel(config)
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

    def _compute_focused_halt_targets(
        self,
        step_hidden_states: tuple[torch.Tensor, ...],
        halt_logits_tensors: tuple[torch.Tensor, ...],
        labels: torch.LongTensor,
        *,
        ignore_index: int,
        min_halt_depth: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not step_hidden_states or not halt_logits_tensors:
            raise ValueError("Focused halt training requires per-step hidden states and halt logits.")
        if len(step_hidden_states) != len(halt_logits_tensors):
            raise ValueError("Focused halt supervision received mismatched step states and halt logits.")

        shifted_labels = torch.cat(
            (labels[..., 1:], torch.full_like(labels[:, :1], ignore_index)),
            dim=1,
        )
        valid_mask = shifted_labels.ne(ignore_index)
        sampled_mask = valid_mask
        max_focus_tokens = self.config.focused_halt_num_tokens
        if max_focus_tokens > 0:
            sampled_mask = torch.zeros_like(valid_mask)
            for batch_idx in range(valid_mask.shape[0]):
                valid_positions = valid_mask[batch_idx].nonzero(as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue
                if valid_positions.numel() > max_focus_tokens:
                    sample_indices = torch.linspace(
                        0,
                        valid_positions.numel() - 1,
                        steps=max_focus_tokens,
                        device=valid_positions.device,
                    ).round().long()
                    valid_positions = valid_positions.index_select(0, sample_indices)
                sampled_mask[batch_idx, valid_positions] = True
        else:
            sampled_mask = valid_mask

        sample_index_pairs = sampled_mask.nonzero(as_tuple=False)
        if sample_index_pairs.numel() == 0:
            raise ValueError("Focused halt supervision requires at least one valid training token.")
        sample_batch_ids = sample_index_pairs[:, 0]
        sampled_labels = shifted_labels[sampled_mask]
        sampled_counts = torch.zeros(
            shifted_labels.shape[0],
            device=labels.device,
            dtype=torch.float32,
        ).scatter_add_(
            0,
            sample_batch_ids,
            torch.ones(sample_batch_ids.shape[0], device=labels.device, dtype=torch.float32),
        )
        step_losses: list[torch.Tensor] = []
        with torch.no_grad():
            for step_hidden in step_hidden_states:
                sampled_hidden = step_hidden.detach()[sampled_mask]
                normed_hidden = self.model.norm(sampled_hidden)
                step_logits = self.lm_head(normed_hidden).float()
                token_loss = F.cross_entropy(
                    step_logits,
                    sampled_labels,
                    reduction="none",
                )
                sample_loss = torch.zeros(
                    shifted_labels.shape[0],
                    device=token_loss.device,
                    dtype=token_loss.dtype,
                ).scatter_add_(0, sample_batch_ids, token_loss)
                sample_loss = sample_loss / sampled_counts.to(device=token_loss.device, dtype=token_loss.dtype).clamp_min(1.0)
                step_losses.append(sample_loss)

        step_loss_tensor = torch.stack(step_losses, dim=0)
        next_step_loss = torch.cat([step_loss_tensor[1:], step_loss_tensor[-1:]], dim=0)
        improvement = step_loss_tensor - next_step_loss

        margin = self.config.focused_halt_improvement_margin
        temperature = max(self.config.focused_halt_target_temperature, 1e-4)
        targets = torch.sigmoid((margin - improvement) / temperature)
        targets[-1] = 1.0
        if min_halt_depth is not None:
            blocked_positions = min(max(int(min_halt_depth) - 1, 0), targets.shape[0])
            if blocked_positions > 0:
                targets[:blocked_positions] = 0.0

        halt_logits = torch.stack(halt_logits_tensors, dim=0)
        halt_loss = F.binary_cross_entropy_with_logits(halt_logits, targets, reduction="none")
        if min_halt_depth is not None:
            target_mask = torch.ones_like(halt_loss)
            if blocked_positions > 0:
                target_mask[:blocked_positions] = 0.0
            focused_halt_loss = (halt_loss * target_mask).sum() / target_mask.sum().clamp_min(1.0)
        else:
            focused_halt_loss = halt_loss.mean()
        return focused_halt_loss, targets.mean(), improvement.mean()

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
        ponder_loss_weight_override: float | None = None,
        focused_halt_loss_weight_override: float | None = None,
        min_halt_depth: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | ReLoopCausalLMOutputWithPast:
        if labels is not None and use_cache is None:
            use_cache = False
        focused_halt_loss_weight = (
            self.config.focused_halt_loss_weight
            if focused_halt_loss_weight_override is None
            else focused_halt_loss_weight_override
        )
        multi_exit_loss_weight = float(self.config.multi_exit_loss_weight)
        stochastic_exit_loss = bool(self.config.stochastic_exit_loss)
        collect_loop_step_states = bool(labels is not None and focused_halt_loss_weight > 0)
        collect_multi_exit = bool(
            labels is not None
            and self.training
            and (multi_exit_loss_weight > 0 or stochastic_exit_loss)
        )
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
            min_halt_depth=min_halt_depth,
            collect_loop_step_states=collect_loop_step_states,
            collect_multi_exit_states=collect_multi_exit,
            **kwargs,
        )

        hidden_states = outputs[0]
        if self.config.fuse_linear_cross_entropy:
            logits = None
        else:
            logits_input = hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:]
            logits = self.lm_head(logits_input)

        loss = None
        focused_halt_loss_tensor = None
        focused_halt_target_mean_tensor = None
        focused_halt_improvement_mean_tensor = None
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

            ponder_loss_weight = (
                self.config.ponder_loss_weight
                if ponder_loss_weight_override is None
                else ponder_loss_weight_override
            )
            if ponder_loss_weight > 0:
                ponder_cost_tensor = getattr(outputs, "ponder_cost_tensor", None)
                if ponder_cost_tensor is not None:
                    loss = loss + ponder_loss_weight * ponder_cost_tensor
            if collect_loop_step_states:
                step_hidden_states = getattr(outputs, "loop_step_hidden_states", None)
                halt_logits_tensors = getattr(outputs, "halt_logits_tensors", None)
                if step_hidden_states is not None and halt_logits_tensors is not None:
                    (
                        focused_halt_loss_tensor,
                        focused_halt_target_mean_tensor,
                        focused_halt_improvement_mean_tensor,
                    ) = self._compute_focused_halt_targets(
                        step_hidden_states=step_hidden_states,
                        halt_logits_tensors=halt_logits_tensors,
                        labels=original_labels,
                        ignore_index=criterion.ignore_index,
                        min_halt_depth=min_halt_depth,
                    )
                    loss = loss + focused_halt_loss_weight * focused_halt_loss_tensor

            # ── Multi-exit loss (legacy, kept for backward compat) ──
            if self.training and multi_exit_loss_weight > 0:
                me_states = getattr(outputs, "multi_exit_hidden_states", None)
                if me_states is not None and len(me_states) > 0:
                    me_num_tokens = 128
                    valid_positions = labels.ne(criterion.ignore_index).nonzero(as_tuple=False)
                    if valid_positions.shape[0] > me_num_tokens:
                        me_idx = torch.randperm(
                            valid_positions.shape[0], device=labels.device
                        )[:me_num_tokens]
                        valid_positions = valid_positions[me_idx]
                    me_batch_idx = valid_positions[:, 0]
                    me_seq_idx = valid_positions[:, 1]
                    me_labels = labels[me_batch_idx, me_seq_idx]
                    me_loss_sum = hidden_states.new_zeros(())
                    for step_h in me_states:
                        sampled_h = step_h[me_batch_idx, me_seq_idx]
                        normed_h = self.model.norm(sampled_h)
                        step_logits = self.lm_head(normed_h)
                        step_loss = F.cross_entropy(step_logits, me_labels)
                        me_loss_sum = me_loss_sum + step_loss
                    loss = loss + multi_exit_loss_weight * (me_loss_sum / len(me_states))

            # ── Stochastic exit loss ──
            # Replace multi-exit's "please all depths equally" with a single
            # randomly-sampled exit per step.  With probability
            # `stochastic_exit_full_depth_prob` the sampled exit is the full
            # depth (main LM loss unchanged).  Otherwise, we sample an
            # intermediate depth and add a subsampled LM loss there.
            # This avoids the multi-exit anti-incentive that makes every
            # depth produce identical output.
            if self.training and bool(self.config.stochastic_exit_loss):
                me_states = getattr(outputs, "multi_exit_hidden_states", None)
                if me_states is not None and len(me_states) > 1:
                    full_depth_prob = float(self.config.stochastic_exit_full_depth_prob)
                    n_max = len(me_states)
                    n_min = max(1, int(self.config.stochastic_exit_min_depth))
                    # Sample exit index: full depth with probability
                    # `full_depth_prob`, else uniform over [n_min, n_max-1].
                    if torch.rand(1).item() < full_depth_prob or n_max <= n_min:
                        exit_idx = n_max - 1  # full depth, already in main loss
                        # Nothing extra to add — main `loss` is the full-depth LM loss
                    else:
                        exit_idx = int(torch.randint(n_min - 1, n_max - 1, (1,)).item())
                        # Subsample tokens for cheap intermediate loss
                        se_num_tokens = 128
                        valid_positions = labels.ne(criterion.ignore_index).nonzero(as_tuple=False)
                        if valid_positions.shape[0] > se_num_tokens:
                            se_idx = torch.randperm(
                                valid_positions.shape[0], device=labels.device
                            )[:se_num_tokens]
                            valid_positions = valid_positions[se_idx]
                        se_batch_idx = valid_positions[:, 0]
                        se_seq_idx = valid_positions[:, 1]
                        se_labels = labels[se_batch_idx, se_seq_idx]
                        step_h = me_states[exit_idx]
                        sampled_h = step_h[se_batch_idx, se_seq_idx]
                        normed_h = self.model.norm(sampled_h)
                        step_logits = self.lm_head(normed_h)
                        se_loss = F.cross_entropy(step_logits, se_labels)
                        # Weight proportional to prob of sampling this exit so
                        # expectation matches a uniform multi-exit loss
                        stochastic_weight = (1.0 - full_depth_prob)
                        loss = loss + stochastic_weight * se_loss

            # ── Soft ponder cost ──
            # λ · E[depth].  E[depth] is the differentiable expected depth
            # computed from per-token AttnRes halt probabilities; gradient
            # flows back to the routing pseudo-queries, giving them
            # incentive to halt earlier on samples that don't need the
            # full depth.  Requires `attnres_halt_per_token=True`.
            ponder_cost_weight = float(self.config.ponder_cost_weight)
            if self.training and ponder_cost_weight > 0:
                ped = getattr(outputs, "ponder_expected_depth_tensor", None)
                if ped is not None:
                    loss = loss + ponder_cost_weight * ped

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ReLoopCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            routing_info=outputs.routing_info if return_routing_info else None,
            ponder_cost_tensor=getattr(outputs, "ponder_cost_tensor", None),
            expected_depth_tensor=getattr(outputs, "expected_depth_tensor", None),
            exit_kl_tensor=getattr(outputs, "exit_kl_tensor", None),
            exit_entropy_tensor=getattr(outputs, "exit_entropy_tensor", None),
            early_exit_mass_tensor=getattr(outputs, "early_exit_mass_tensor", None),
            routing_entropy_tensor=getattr(outputs, "routing_entropy_tensor", None),
            focused_halt_loss_tensor=focused_halt_loss_tensor,
            focused_halt_target_mean_tensor=focused_halt_target_mean_tensor,
            focused_halt_improvement_mean_tensor=focused_halt_improvement_mean_tensor,
        )
