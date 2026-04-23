"""starVLA integration for block-level AttnRes retrofit (per-block in-backbone).

Port of Part-2 retrofit (retrofit/qwen3vl_attnres_retrofit.py) into starVLA's
QwenOFT pipeline. Previously this module exposed an OBSERVER adapter that only
injected the last-block AttnRes correction into ``last_hidden`` — wasting 13/14
router+adapter parameters. This version correctly runs per-block AttnRes
*in-backbone*: for each block n,

    r_n        = Σ α_{i→n} · h_i        (router over completed block outputs)
    x_n        = h_{n-1} + γ_n · A_n(r_n - h_{n-1})
    h_n        = Block_n(x_n)             (actual transformer layers)

matching Part 2 retrofit's forward structure exactly.

starVLA interface
-----------------
QwenOFT / QwenGR00T frameworks consume:

  ``build_starvla_attnres_adapter(config, hidden_size, num_hidden_layers)``
      returns the adapter (router + adapters + γ + schedule state). No forward
      logic here anymore — it's just a parameter container.

  ``StarVLABackboneSkipContext``
      context manager that monkey-patches ``text_model.forward`` with per-block
      AttnRes. Entered once per ``forward_with_attnres_skip`` call in QWen3.py.
      Exposes ``get_summary()`` returning the routing_info dict.
"""
from __future__ import annotations

import contextlib
import math
import types
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers (ported verbatim from retrofit/qwen3vl_attnres_retrofit.py)
# ---------------------------------------------------------------------------

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
        values = torch.stack(completed_outputs, dim=0)   # [N, B, T, H]
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
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


# ---------------------------------------------------------------------------
# Adapter: parameter container (router + adapters + γ + curriculum)
# ---------------------------------------------------------------------------

class StarVLAAttnResAdapter(nn.Module):
    """Parameter container consumed by ``StarVLABackboneSkipContext``.

    Exposes the router / adapters / γ that the per-block forward uses; no
    forward method is intended to be called from outside the context. The
    attnres_adapter attribute on QwenOFT / QwenGR00T holds this object so it
    participates in state_dict (trainable) and so ``load_retrofit_state`` can
    warm-start from Part-2 retrofit checkpoints.
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        n_blocks: int = 14,
        adapter_rank: int = 256,
        temperature: float = 1.0,
        initializer_range: float = 0.02,
        no_adapter: bool = False,
    ):
        super().__init__()
        if num_hidden_layers % n_blocks != 0:
            raise ValueError(
                f"num_hidden_layers={num_hidden_layers} must be divisible by n_blocks={n_blocks}"
            )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_blocks = n_blocks
        self.layers_per_block = num_hidden_layers // n_blocks
        self.no_adapter = no_adapter

        self.router = BlockAttnResRouter(
            hidden_size=hidden_size,
            num_sources=n_blocks + 1,
            temperature=temperature,
            use_positional_bias=True,
            initializer_range=initializer_range,
        )
        if no_adapter:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(n_blocks)])
        else:
            self.adapters = nn.ModuleList(
                [ResidualAdapter(hidden_size, adapter_rank=adapter_rank) for _ in range(n_blocks)]
            )
        # γ: learnable. Optimizer drives it; curriculum uses a separate buffer
        # so ZeRO-2's master-weight all-gather doesn't undo our override.
        self.gamma = nn.Parameter(torch.ones(n_blocks))

        # γ-curriculum (scale ramps 0 → target over `_gamma_ramp_steps`).
        self._gamma_schedule_enabled: bool = False
        self._gamma_ramp_steps: int = 0
        self._gamma_target: float = 1.0
        # Non-persistent buffers — not saved, so loading legacy ckpts works.
        self.register_buffer("_gamma_scale", torch.ones(n_blocks), persistent=False)
        self.register_buffer("_train_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def enable_gamma_curriculum(self, ramp_steps: int, target: float = 1.0):
        self._gamma_schedule_enabled = True
        self._gamma_ramp_steps = int(max(1, ramp_steps))
        self._gamma_target = float(target)
        self._train_step.zero_()
        self._gamma_scale.zero_()

    def advance_curriculum(self):
        """Called once per training forward by the skip-context; updates scale."""
        if self._gamma_schedule_enabled and self.training:
            step = int(self._train_step.item())
            frac = min(1.0, step / max(1, self._gamma_ramp_steps))
            self._gamma_scale.fill_(frac * self._gamma_target)
            self._train_step += 1

    def effective_gamma(self, block_idx: int) -> torch.Tensor:
        """Learnable γ × curriculum scale (scale=1 when no schedule)."""
        return self.gamma[block_idx] * self._gamma_scale[block_idx]

    # --- Load Part-2 retrofit warm-start state ----------------------------

    def load_retrofit_state(self, state_path: str, strict: bool = False) -> None:
        ck = torch.load(state_path, map_location="cpu")
        cfg = ck.get("config", {})
        if cfg.get("num_blocks", self.n_blocks) != self.n_blocks:
            raise ValueError(
                f"state n_blocks={cfg.get('num_blocks')} != adapter n_blocks={self.n_blocks}"
            )
        self.router.load_state_dict(ck["router"], strict=strict)
        if not self.no_adapter and ck.get("adapters"):
            self.adapters.load_state_dict(ck["adapters"], strict=strict)
        self.gamma.data.copy_(ck["gamma"])


# ---------------------------------------------------------------------------
# Per-block AttnRes backbone patch (the real integration)
# ---------------------------------------------------------------------------

class StarVLABackboneSkipContext(contextlib.AbstractContextManager):
    """Monkey-patch the Qwen3-VL language_model forward to run per-block AttnRes.

    Port of retrofit/qwen3vl_attnres_retrofit.py's ``_text_model_forward``
    mapped to starVLA's QWen3 interface. Active only inside the ``with`` block;
    restores original forward on exit.

    Skip behaviour: with ``enable_skipping=False`` (default), all blocks run;
    we still run per-block AttnRes so the adapter/router parameters are
    actually in the forward graph. With skip on, the ``dynamic_skip_config``
    dict is consulted per block following the retrofit / Part-1 schema:

        {"thresholds": {block_idx: τ_n},     # per-block w_recent quantile
         "eligible_blocks": set[int] | None,  # None = all blocks eligible
         "max_skips": int | None}             # cap per forward, None = no cap

    Cache semantics: when ``past_key_values`` is provided, skipped blocks
    still populate each inner layer's KV cache via a K/V-only path
    (``input_layernorm → k_proj/v_proj → k_norm → RoPE → cache.update``),
    so a subsequent decode step sees a consistent-length cache. Q-proj,
    attention, o_proj and the MLP are fully skipped.
    """

    def __init__(
        self,
        adapter: StarVLAAttnResAdapter,
        *,
        enable_skipping: bool = False,
        dynamic_skip_config: Optional[dict] = None,
    ):
        self.adapter = adapter
        self.enable_skipping = bool(enable_skipping)
        self.dynamic_skip_config = dynamic_skip_config
        # Populated after forward
        self._summary: dict = {
            "flops_ratio": 1.0,
            "effective_block_ratio": 1.0,
            "num_blocks_executed": float(adapter.n_blocks),
            "backbone_compute_preserved": True,
            "skipped_blocks": [],
            "keep_mask": None,
        }
        # Owning text_model reference for patch/unpatch
        self._text_model = None
        self._orig_forward = None

    # ----- Context manager machinery --------------------------------------

    def bind_text_model(self, text_model):
        """Called by QWen3.forward_with_attnres_skip before entering the context.

        IMPORTANT: we MUST NOT assign text_model as a plain attribute on the
        adapter (``self.adapter._bound_text_model = text_model``) because
        ``nn.Module.__setattr__`` would auto-register it as a child submodule
        and the full 2B text_model weights would end up duplicated under
        ``attnres_adapter._bound_text_model.*`` in the state_dict. Store the
        reference on the context only; the adapter doesn't need it.
        """
        self._text_model = text_model

    def __enter__(self):
        if self._text_model is None:
            raise RuntimeError(
                "StarVLABackboneSkipContext: call bind_text_model() before entering"
            )
        tm = self._text_model
        self._orig_forward = tm.forward
        controller = self
        tm.forward = types.MethodType(
            lambda self_tm, *args, **kwargs: controller._patched_forward(
                self_tm, *args, **kwargs
            ),
            tm,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore the original forward
        if self._text_model is not None and self._orig_forward is not None:
            self._text_model.forward = self._orig_forward
        return False

    def get_summary(self) -> dict:
        return dict(self._summary)

    # ----- The patched forward (ported from Part 2) ---------------------------

    def _patched_forward(
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

        adapter = self.adapter
        # Per-step curriculum advance (no-op outside train mode)
        adapter.advance_curriculum()

        # Resolve dynamic-skip config once per forward.
        dyn_cfg = self.dynamic_skip_config if self.enable_skipping else None
        thr_map: dict = (dyn_cfg or {}).get("thresholds", {}) or {}
        eligible = (dyn_cfg or {}).get("eligible_blocks")
        max_skips = (dyn_cfg or {}).get("max_skips")

        completed: list[torch.Tensor] = [inputs_embeds]
        prev_block = inputs_embeds
        alpha_list: list[torch.Tensor] = []
        skipped_blocks: list[int] = []
        dyn_skipped_count = 0
        layer_counter = 0

        for block_idx in range(adapter.n_blocks):
            # ----- Phase 1: compute routed state + correction -----------
            routed, alpha = adapter.router.route(block_idx + 1, completed)
            alpha_list.append(alpha)
            delta = routed - prev_block
            correction = adapter.adapters[block_idx](delta) if not adapter.no_adapter else delta
            gamma_n = adapter.effective_gamma(block_idx).to(correction.dtype)
            block_input = prev_block + gamma_n * correction

            # ----- Skip decision (recent_weight_gt, per-block τ) --------
            should_skip = False
            if dyn_cfg is not None and alpha is not None and alpha.shape[-1] >= 2:
                thr = thr_map.get(block_idx)
                if (
                    thr is not None
                    and (eligible is None or block_idx in eligible)
                    and (max_skips is None or dyn_skipped_count < max_skips)
                ):
                    # alpha shape [B, T, N]; last source dim = most recent completed block.
                    # Averaging over (B, T) matches retrofit Part-2 and Part-1 exactly.
                    w_recent = float(alpha[..., -1].float().mean().item())
                    if w_recent > float(thr):
                        should_skip = True
                        dyn_skipped_count += 1
                        skipped_blocks.append(block_idx)

            # ----- Phase 2: run (or skip) the block's layers ------------
            if should_skip:
                h = block_input
                # Maintain KV cache consistency: run only the layers' K/V
                # projection path on block_input so the cache keeps its
                # full length. Q-proj, attention, o_proj and MLP are
                # entirely skipped (≈ 90 % layer compute saved on GQA-16).
                if past_key_values is not None:
                    for l_off in range(adapter.layers_per_block):
                        layer = text_model.layers[layer_counter + l_off]
                        attn = layer.self_attn
                        normed = layer.input_layernorm(h)
                        input_shape = normed.shape[:-1]
                        hidden_shape = (*input_shape, -1, attn.head_dim)
                        k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
                        v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
                        cos, sin = position_embeddings
                        # apply_rotary_pos_emb is symmetric in q/k; pass k twice.
                        _, k = apply_rotary_pos_emb(k, k, cos, sin)
                        past_key_values.update(
                            k, v, attn.layer_idx,
                            {"sin": sin, "cos": cos, "cache_position": cache_position},
                        )
                layer_counter += adapter.layers_per_block
            else:
                h = block_input
                for _ in range(adapter.layers_per_block):
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

        hidden_states = text_model.norm(completed[-1])

        # Summary
        n_exec = adapter.n_blocks - len(skipped_blocks)
        self._summary = {
            "flops_ratio": n_exec / adapter.n_blocks,
            "effective_block_ratio": n_exec / adapter.n_blocks,
            "num_blocks_executed": float(n_exec),
            "backbone_compute_preserved": len(skipped_blocks) == 0,
            "skipped_blocks": skipped_blocks,
            "keep_mask": [b not in skipped_blocks for b in range(adapter.n_blocks)],
        }

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_starvla_attnres_adapter(
    config: Any,
    hidden_size: int,
    num_hidden_layers: int,
) -> Optional[Any]:
    """Build an AttnRes parameter container if ``framework.attnres.enabled``."""
    try:
        enabled = bool(config.framework.attnres.enabled)
    except Exception:
        enabled = False
    if not enabled:
        return None

    attncfg = config.framework.attnres
    n_blocks = int(getattr(attncfg, "n_blocks", 14))
    adapter_rank = int(getattr(attncfg, "adapter_rank", 256))
    temperature = float(getattr(attncfg, "temperature", 1.0))
    no_adapter = bool(getattr(attncfg, "no_adapter", False))
    init_state = getattr(attncfg, "init_state_path", None)
    gamma_ramp_steps = int(getattr(attncfg, "gamma_ramp_steps", 0))
    gamma_target = float(getattr(attncfg, "gamma_target", 1.0))

    adapter = StarVLAAttnResAdapter(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        n_blocks=n_blocks,
        adapter_rank=adapter_rank,
        temperature=temperature,
        no_adapter=no_adapter,
    )
    if init_state:
        adapter.load_retrofit_state(init_state, strict=False)
    if gamma_ramp_steps > 0:
        adapter.enable_gamma_curriculum(gamma_ramp_steps, gamma_target)
    return adapter
