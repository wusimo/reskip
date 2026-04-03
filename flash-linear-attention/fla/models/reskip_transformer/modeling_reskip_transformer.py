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


@dataclass
class ReSkipBaseModelOutputWithPast(BaseModelOutputWithPast):
    routing_info: dict[str, Any] | None = None


@dataclass
class ReSkipCausalLMOutputWithPast(CausalLMOutputWithPast):
    routing_info: dict[str, Any] | None = None


class DepthAttentionResidual(nn.Module):
    def __init__(self, config: ReSkipTransformerConfig):
        super().__init__()
        self.temperature = config.attn_res_temperature
        self.w_query = nn.Parameter(torch.empty(config.hidden_size))
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def reset_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(self.w_query, mean=0.0, std=initializer_range)
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=initializer_range)

    def forward(
        self,
        source_states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if len(source_states) == 0:
            raise ValueError("DepthAttentionResidual needs at least one source state.")
        if len(source_states) == 1:
            routed = source_states[0]
            weights = routed.new_ones(routed.shape[0], routed.shape[1], 1)
            return routed, weights if return_weights else None

        sources = torch.stack(source_states, dim=2)
        keys = self.key_proj(sources)
        scores = torch.einsum("d,btnd->btn", self.w_query, keys)
        scores = scores / (math.sqrt(sources.shape[-1]) * self.temperature)
        weights = torch.softmax(scores, dim=-1)
        routed = torch.sum(weights.unsqueeze(-1) * sources, dim=2)
        return routed, weights if return_weights else None


class ReSkipTransformerBlock(GradientCheckpointingLayer):
    def __init__(self, config: ReSkipTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
        )
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
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, torch.Tensor | None, Cache | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


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
            [
                ReSkipTransformerBlock(config, first_layer_idx + offset)
                for offset in range(layers_per_block)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        attentions = None
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                attentions = layer_outputs[1]
            if use_cache:
                past_key_values = layer_outputs[2 if output_attentions else 1]
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


class ReSkipTransformerPreTrainedModel(PreTrainedModel):
    config_class = ReSkipTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReSkipBlockGroup", "ReSkipTransformerBlock"]
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
        elif isinstance(module, DepthAttentionResidual):
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
        self.depth_routers = nn.ModuleList(
            [DepthAttentionResidual(config) for _ in range(self.num_block_positions)]
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.gradient_checkpointing = False
        self._skip_keep_mask = self._normalize_keep_mask(config.skip_keep_mask)
        self._last_routing_info: dict[str, Any] | None = None

        self.post_init()

    def _build_block_schedule(self) -> list[int]:
        if not self.config.enable_looping:
            return list(range(self.config.attn_res_num_blocks))
        return [
            position % self.config.num_recurrent_blocks
            for position in range(self.config.attn_res_num_blocks)
        ]

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

    def _compute_importance_matrix(
        self,
        routing_weights: list[torch.Tensor],
    ) -> torch.Tensor:
        device = routing_weights[0].device if routing_weights else self.embeddings.weight.device
        matrix = torch.zeros(
            self.num_block_positions,
            self.num_block_positions,
            device=device,
            dtype=torch.float32,
        )
        for target_pos, weights in enumerate(routing_weights):
            avg_weights = weights.mean(dim=(0, 1)).float()
            for source_pos in range(target_pos):
                source_index = source_pos + 1
                if source_index < avg_weights.shape[0]:
                    matrix[source_pos, target_pos] = avg_weights[source_index]
        return matrix

    def _build_routing_info(
        self,
        routing_weights: list[torch.Tensor],
        blocks_executed: list[tuple[int, int, int]],
        keep_mask: list[bool] | None,
    ) -> dict[str, Any]:
        importance_matrix = self._compute_importance_matrix(routing_weights)
        block_importance = []
        for block_pos in range(self.num_block_positions):
            downstream = importance_matrix[block_pos, block_pos + 1 :]
            block_importance.append(float(downstream.max().item()) if downstream.numel() > 0 else 1.0)
        return {
            "routing_weights": routing_weights,
            "importance_matrix": importance_matrix,
            "block_importance": block_importance,
            "block_schedule": list(self.block_schedule),
            "keep_mask": keep_mask if keep_mask is not None else [True] * self.num_block_positions,
            "blocks_executed": blocks_executed,
            "num_blocks_executed": sum(1 for _, _, status in blocks_executed if status >= 0),
            "effective_depth": sum(1 for _, _, status in blocks_executed if status >= 0),
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

        all_hidden_states = () if output_hidden_states else None
        next_cache = None
        keep_mask = self._resolve_keep_mask(enable_skipping, skip_keep_mask)
        adapted_outputs = [inputs_embeds]
        routing_weights: list[torch.Tensor] = []
        blocks_executed: list[tuple[int, int, int]] = []

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        for position, block_idx in enumerate(self.block_schedule):
            if output_hidden_states:
                all_hidden_states += (adapted_outputs[-1],)

            routed_input, weights = self.depth_routers[position](adapted_outputs, return_weights=True)
            routing_weights.append(weights)
            should_execute = True if keep_mask is None else keep_mask[position]

            if should_execute:
                if self.gradient_checkpointing and self.training:
                    block_outputs = self._gradient_checkpointing_func(
                        self.layers[block_idx].__call__,
                        routed_input,
                        attention_mask,
                        past_key_values,
                        False,
                        use_cache,
                        **kwargs,
                    )
                else:
                    block_outputs = self.layers[block_idx](
                        routed_input,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        output_attentions=False,
                        use_cache=use_cache,
                        **kwargs,
                    )
                adapted_outputs.append(block_outputs[0])
                blocks_executed.append((position, block_idx, 0))
                if use_cache:
                    next_cache = block_outputs[1]
            else:
                blocks_executed.append((position, block_idx, -1))

        hidden_states = self.norm(adapted_outputs[-1])
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        routing_info = self._build_routing_info(routing_weights, blocks_executed, keep_mask)
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
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

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
