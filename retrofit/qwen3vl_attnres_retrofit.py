from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen3VLAttnResRetrofitOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    alpha_list: list[torch.Tensor] | None = None
    skip_trace: list[dict] | None = None
    block_inputs: list[torch.Tensor] | None = None
    block_outputs: list[torch.Tensor] | None = None
    surrogate_outputs: list[torch.Tensor | None] | None = None
    entropy_penalty: torch.Tensor | None = None


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    fp = x.float()
    inv = torch.rsqrt(fp.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (fp * inv).to(x.dtype)


class BlockAttnResRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        temperature: float = 1.0,
        use_positional_bias: bool = True,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = num_sources
        self.base_temperature = temperature
        self.w_query = nn.Parameter(torch.empty(num_sources, hidden_size))
        if use_positional_bias:
            self.key_pos_bias = nn.Parameter(torch.empty(num_sources, hidden_size))
        else:
            self.register_parameter("key_pos_bias", None)
        nn.init.normal_(self.w_query, std=initializer_range)
        if self.key_pos_bias is not None:
            nn.init.normal_(self.key_pos_bias, std=initializer_range)

    def route(self, position: int, completed_outputs: list[torch.Tensor]):
        values = torch.stack(completed_outputs, dim=0)  # [N, B, T, H]
        keys = _rms_norm(values)
        if self.key_pos_bias is not None:
            bias = self.key_pos_bias[:position].to(keys.dtype)
            keys = keys + bias[:, None, None, :]
        query = self.w_query[position].to(keys.dtype)
        scale = math.sqrt(values.shape[-1]) * self.base_temperature
        scores = torch.einsum("h,nbth->nbt", query, keys) / scale
        alpha = torch.softmax(scores.float(), dim=0).to(values.dtype)
        routed = torch.einsum("nbt,nbth->bth", alpha, values)
        return routed, alpha.permute(1, 2, 0)


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_rank: int = 128):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_rank, bias=False)
        self.up = nn.Linear(adapter_rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        # Small random (not zero) — combined with γ=0 we still get identity at
        # init (x_n = prev_block + 0 * adapter = prev_block), but gradients
        # can flow through adapter(delta) into γ. Pure zero-init on both γ AND
        # up.weight causes gradient deadlock — γ stuck at 0 forever.
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class Qwen3VLAttnResRetrofit(nn.Module):
    """Retrofit Qwen3-VL so AttnRes enters the normal forward path safely.

    For selected late blocks we compute:
      routed_n = AttnRes(h_0..h_{n-1})
      x_n = h_{n-1} + gamma_n * Adapter_n(routed_n - h_{n-1})

    Full path:
      h_n = Block_n(x_n)

    Skip path:
      h_n = x_n

    `gamma_n` is zero-initialized so the wrapper starts as the original model.
    """

    def __init__(
        self,
        base_model,
        num_blocks: int = 14,
        skippable_blocks: Iterable[int] | None = None,
        adapter_rank: int = 128,
        initializer_range: float = 0.02,
        no_adapter: bool = False,
    ):
        # no_adapter=True → pure Route A: x_n = (1-γ)h_{n-1} + γ·r_n
        #   (matches Part 1 AttnRes structurally when γ=1)
        # no_adapter=False (default) → x_n = h_{n-1} + γ·Adapter(r_n - h_{n-1})
        super().__init__()
        self.base_model = base_model
        cfg = base_model.config.text_config
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_hidden_layers
        if self.num_layers % num_blocks != 0:
            raise ValueError(
                f"num_hidden_layers={self.num_layers} must be divisible by num_blocks={num_blocks}"
            )
        self.num_blocks = num_blocks
        self.layers_per_block = self.num_layers // num_blocks
        if skippable_blocks is None:
            # Apply AttnRes injection to ALL blocks so the retrofit produces a
            # genuine AttnRes model (not a hybrid where only late blocks went AttnRes).
            # γ=0 zero-init already guarantees identity-at-init, so restricting
            # to late blocks is unnecessary conservatism and weakens the Part-2 claim.
            skippable_blocks = list(range(num_blocks))
        self.skippable_blocks = tuple(sorted(set(int(x) for x in skippable_blocks)))
        self.skippable_block_set = set(self.skippable_blocks)

        self.router = BlockAttnResRouter(
            hidden_size=self.hidden_size,
            num_sources=num_blocks + 1,
            use_positional_bias=True,
            initializer_range=initializer_range,
        )
        self.no_adapter = no_adapter
        if no_adapter:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        else:
            self.adapters = nn.ModuleList(
                [ResidualAdapter(self.hidden_size, adapter_rank=adapter_rank) for _ in range(num_blocks)]
            )
        # Identity-at-init comes from adapter.up == 0, not from gamma == 0.
        # Keeping gamma non-zero allows adapter weights to receive gradient
        # immediately, avoiding the dead-start behavior of gamma=0.
        self.gamma = nn.Parameter(torch.zeros(num_blocks))

        self._fwd_alpha_list: list[torch.Tensor] | None = None
        self._fwd_skip_trace: list[dict] | None = None
        self._fwd_block_inputs: list[torch.Tensor] | None = None
        self._fwd_block_outputs: list[torch.Tensor] | None = None
        self._fwd_surrogate_outputs: list[torch.Tensor | None] | None = None
        self._fwd_entropy: torch.Tensor | None = None
        self._active_skip_blocks: set[int] = set()
        self._dynamic_skip_config: dict | None = None
        self._collect_block_states = False
        self._return_alpha_flag = False

        self._install_retrofit_forward()

    @property
    def text_layers(self):
        return self.base_model.model.language_model.layers

    @property
    def lm_head(self):
        return self.base_model.lm_head

    def _install_retrofit_forward(self):
        retrofit = self
        lm = self.base_model.model.language_model
        if not hasattr(lm, "_original_forward"):
            lm._original_forward = lm.forward

        def patched_forward(
            self_lm,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        ):
            return retrofit._text_model_forward(
                self_lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        lm.forward = types.MethodType(patched_forward, lm)

    def _block_contains_deepstack_layers(self, block_idx: int, deepstack_visual_embeds) -> bool:
        if deepstack_visual_embeds is None:
            return False
        if not hasattr(deepstack_visual_embeds, "__len__"):
            return False
        if len(deepstack_visual_embeds) == 0:
            return False
        start = block_idx * self.layers_per_block
        end = start + self.layers_per_block
        return start < len(deepstack_visual_embeds) and end > 0

    def _compute_block_input(
        self,
        block_idx: int,
        prev_block: torch.Tensor,
        completed: list[torch.Tensor],
        collect_alpha: bool,
        compute_entropy: bool = False,
    ):
        if block_idx not in self.skippable_block_set:
            return prev_block, None, None, None
        routed, alpha = self.router.route(block_idx + 1, completed)
        delta = routed - prev_block
        corrected = prev_block + self.gamma[block_idx].to(prev_block.dtype) * self.adapters[block_idx](delta)
        # Entropy is only consumed as a training regulariser. Computing it on
        # every inference forward wastes ~14 kernel launches with no effect
        # on outputs; gate behind compute_entropy.
        entropy = None
        if compute_entropy:
            entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
        if collect_alpha:
            return corrected, alpha, routed, entropy
        return corrected, None, routed, entropy

    def _text_model_forward(
        self,
        text_model,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
            create_causal_mask,
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=text_model.config)
        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=text_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)

        # Fast-path gate: only the analysis / training hooks need per-block
        # skip_trace (w_recent / top_source / gamma), full alpha_list, and
        # entropy. Plain inference (benchmark, lmms-eval, VLA inference)
        # leaves these off and avoids 3 CUDA syncs + extra kernel launches
        # per block. Enable when the caller actually consumes them.
        collect_trace = (
            self._return_alpha_flag
            or self._collect_block_states
            or self.training
            or (self._dynamic_skip_config is not None)
        )
        compute_entropy_flag = self.training

        completed: list[torch.Tensor] = [inputs_embeds]
        prev_block = inputs_embeds
        alpha_list: list[torch.Tensor] = []
        skip_trace: list[dict] = []
        block_inputs: list[torch.Tensor] = []
        block_outputs: list[torch.Tensor] = []
        surrogate_outputs: list[torch.Tensor | None] = []
        entropy_accum: torch.Tensor | None = None
        layer_counter = 0
        gamma_cpu: list[float] | None = None  # lazy fill only if collect_trace

        dynamic_cfg = self._dynamic_skip_config  # None or dict
        dyn_skipped_count = 0
        # Precompute Python-side dyn-skip config once per forward.
        dyn_thr_map = (dynamic_cfg or {}).get("thresholds", {}) or {}
        dyn_eligible = (dynamic_cfg or {}).get("eligible_blocks")
        dyn_max_skips = (dynamic_cfg or {}).get("max_skips")
        for block_idx in range(self.num_blocks):
            # alpha is needed only for trace or dyn-skip; routed+corrected
            # always needed for the forward math.
            need_alpha = collect_trace or (dynamic_cfg is not None)
            block_input, alpha, routed, entropy = self._compute_block_input(
                block_idx, prev_block, completed,
                collect_alpha=need_alpha,
                compute_entropy=compute_entropy_flag,
            )
            if entropy is not None:
                entropy_accum = entropy if entropy_accum is None else entropy_accum + entropy
            if alpha is not None and collect_trace:
                alpha_list.append(alpha)

            skip_requested = block_idx in self._active_skip_blocks
            # Dyn-skip decision: per-block τ, eligible mask, max_skips cap.
            # Only sync w_recent to CPU when the Python-side gating passes,
            # so retrofit-full (no dyn cfg) and non-eligible blocks pay zero
            # per-block sync.
            dynamic_skip_requested = False
            if (
                dynamic_cfg is not None
                and alpha is not None
                and alpha.shape[-1] >= 2
            ):
                thr = dyn_thr_map.get(block_idx)
                if (
                    thr is not None
                    and (dyn_eligible is None or block_idx in dyn_eligible)
                    and (dyn_max_skips is None or dyn_skipped_count < dyn_max_skips)
                ):
                    if bool((alpha[..., -1].mean().float() > float(thr)).item()):
                        dynamic_skip_requested = True

            deepstack_sensitive = self._block_contains_deepstack_layers(block_idx, deepstack_visual_embeds)
            should_skip = (
                (skip_requested or dynamic_skip_requested)
                and block_idx in self.skippable_block_set
                and not deepstack_sensitive
            )
            if dynamic_skip_requested and should_skip:
                dyn_skipped_count += 1

            if block_input is None:
                block_input = prev_block
                if collect_trace:
                    surrogate_outputs.append(None)
            elif collect_trace:
                surrogate_outputs.append(block_input)

            if should_skip:
                h = block_input
                # Maintain KV-cache consistency for the skipped layers so
                # subsequent decode steps see a full-length per-layer cache.
                # We run only the layer's K/V path (input_layernorm →
                # k_proj/v_proj → k_norm → RoPE → cache.update) on the same
                # block_input; Q/attention/output-proj/MLP are fully skipped.
                # Correctness intuition: a skipped block is treated as an
                # identity transformation, so every layer inside it sees the
                # same block_input as its hidden state input.
                if past_key_values is not None:
                    for l_off in range(self.layers_per_block):
                        layer = text_model.layers[layer_counter + l_off]
                        attn = layer.self_attn
                        normed = layer.input_layernorm(h)
                        input_shape = normed.shape[:-1]
                        hidden_shape = (*input_shape, -1, attn.head_dim)
                        k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
                        v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
                        cos, sin = position_embeddings
                        # apply_rotary_pos_emb is symmetric in q/k; pass k
                        # twice and keep the rotated k. Cheap (one rotation).
                        _, k = apply_rotary_pos_emb(k, k, cos, sin)
                        past_key_values.update(
                            k, v, attn.layer_idx,
                            {"sin": sin, "cos": cos, "cache_position": cache_position},
                        )
                layer_counter += self.layers_per_block
            else:
                h = block_input
                for _ in range(self.layers_per_block):
                    layer = text_model.layers[layer_counter]
                    h = layer(
                        h,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                    if isinstance(h, tuple):
                        h = h[0]
                    if (
                        deepstack_visual_embeds is not None
                        and layer_counter < len(deepstack_visual_embeds)
                    ):
                        h = text_model._deepstack_process(
                            h, visual_pos_masks, deepstack_visual_embeds[layer_counter]
                        )
                    layer_counter += 1

            prev_block = h
            completed.append(h)
            if self._collect_block_states:
                block_inputs.append(block_input)
                block_outputs.append(h)
            if collect_trace:
                # All per-block sync / bookkeeping gated here so plain
                # inference does not pay 3 CUDA syncs + 2 kernel launches
                # per block. w_recent / top_source / gamma are only of
                # interest to calibration + analysis callers.
                if gamma_cpu is None:
                    gamma_cpu = self.gamma.detach().float().cpu().tolist()  # one sync
                if alpha is not None and alpha.shape[-1] >= 2:
                    w_recent_val = float(alpha[..., -1].float().mean().item())
                    top_source = int(alpha.float().mean(dim=(0, 1)).argmax().item())
                else:
                    w_recent_val = None
                    top_source = None
                skip_trace.append(
                    {
                        "block_idx": block_idx,
                        "used_attnres": block_idx in self.skippable_block_set,
                        "skipped": should_skip,
                        "skip_requested": skip_requested,
                        "dynamic_skip_requested": dynamic_skip_requested,
                        "w_recent": w_recent_val,
                        "deepstack_sensitive": deepstack_sensitive,
                        "gamma": gamma_cpu[block_idx],
                        "top_source": top_source,
                    }
                )

        hidden_states = text_model.norm(completed[-1])
        self._fwd_alpha_list = alpha_list if collect_trace else None
        self._fwd_skip_trace = skip_trace if collect_trace else None
        self._fwd_block_inputs = block_inputs if self._collect_block_states else None
        self._fwd_block_outputs = block_outputs if self._collect_block_states else None
        self._fwd_surrogate_outputs = surrogate_outputs if self._collect_block_states else None
        self._fwd_entropy = entropy_accum
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        return_alpha: bool = False,
        return_block_states: bool = False,
        skip_block_indices: Iterable[int] | None = None,
        dynamic_skip_config: dict | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Qwen3VLAttnResRetrofitOutput:
        """dynamic_skip_config: dict with keys:
            thresholds: {block_idx: float τ_n}  — from phase-1 w_recent calibration
            eligible_blocks: set of int | None  — if None, all skippable allowed
            max_skips: int | None  — cap on blocks skipped per input

        use_cache: passes through to base_model. Skipped blocks still update
            each inner layer's KV cache by running only the cheap K/V
            projection path on block_input (treating the skipped block as an
            identity for its output), so subsequent decode steps can attend
            over a consistent-length cache.
        """
        self._fwd_alpha_list = None
        self._fwd_skip_trace = None
        self._fwd_block_inputs = None
        self._fwd_block_outputs = None
        self._fwd_surrogate_outputs = None
        self._fwd_entropy = None
        self._active_skip_blocks = set(int(x) for x in (skip_block_indices or []))
        self._dynamic_skip_config = dynamic_skip_config
        self._collect_block_states = return_block_states
        self._return_alpha_flag = return_alpha

        out = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,
            use_cache=use_cache,
            **kwargs,
        )
        logits = out.logits

        loss = None
        if labels is not None:
            shifted = torch.cat(
                [labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1
            )
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), shifted.view(-1), ignore_index=-100
            )

        return Qwen3VLAttnResRetrofitOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=getattr(out, "hidden_states", None),
            alpha_list=self._fwd_alpha_list if return_alpha else None,
            skip_trace=self._fwd_skip_trace,
            block_inputs=self._fwd_block_inputs,
            block_outputs=self._fwd_block_outputs,
            surrogate_outputs=self._fwd_surrogate_outputs,
            entropy_penalty=self._fwd_entropy,
        )

    def freeze_vision(self):
        for p in self.base_model.model.visual.parameters():
            p.requires_grad = False

    def freeze_base(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

    def retrofit_parameters(self):
        return list(self.router.parameters()) + list(self.adapters.parameters()) + [self.gamma]

    def late_block_parameters(self):
        params = []
        for block_idx in self.skippable_blocks:
            start = block_idx * self.layers_per_block
            end = start + self.layers_per_block
            for layer_idx in range(start, end):
                params.extend(list(self.text_layers[layer_idx].parameters()))
        return params

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
