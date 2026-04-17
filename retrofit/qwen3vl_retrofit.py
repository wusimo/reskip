"""Retrofit Qwen3-VL-2B's text decoder with AttnRes routing.

Applies Route A (interpolation gate, β=0.1) + positional bias on keys
+ entropy regularizer — the combo that proved to learn sparse α at 74M.

Only the LANGUAGE MODEL's decoder layers are retrofitted; vision encoder
is untouched. Block-level AttnRes with N blocks (default N=14, so each
block = 2 consecutive text layers for Qwen3-VL's 28 layers).

Usage (text-only training):
    python qwen3vl_retrofit.py --num-blocks 14 --tokens 200_000_000 ...

The wrapper surfaces a .forward(input_ids, labels, ...) that mimics a
causal-LM interface for fineweb-style text fine-tune. Vision inputs can
be attached later for VLA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


@dataclass
class Qwen3VLRetrofitOutput:
    last_hidden_state: torch.Tensor
    loss: torch.Tensor | None = None
    entropy_penalty: torch.Tensor | None = None
    alpha_list: list[torch.Tensor] | None = None


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    fp = x.float()
    inv = torch.rsqrt(fp.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (fp * inv).to(x.dtype)


class BlockAttnResQwen(nn.Module):
    """AttnRes router for Qwen3-VL's text backbone (GQA, RoPE-based)."""

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        temperature: float = 1.0,
        use_positional_bias: bool = True,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.base_temperature = temperature
        self.use_positional_bias = use_positional_bias

        # pseudo-queries per position (index 0..num_blocks; 0 = embedding)
        self.w_query = nn.Parameter(torch.empty(num_blocks, hidden_size))
        if use_positional_bias:
            self.key_pos_bias = nn.Parameter(torch.empty(num_blocks, hidden_size))
        else:
            self.register_parameter("key_pos_bias", None)
        self._init(initializer_range)

    def _init(self, initializer_range: float):
        nn.init.normal_(self.w_query, mean=0.0, std=initializer_range)
        if self.key_pos_bias is not None:
            nn.init.normal_(self.key_pos_bias, mean=0.0, std=initializer_range)

    def route(
        self,
        position: int,
        completed_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routed output at position (>=1) over completed sources."""
        assert position >= 1 and len(completed_outputs) == position
        values = torch.stack(completed_outputs, dim=0)  # [n, B, T, H]
        keys = _rms_norm(values)
        if self.key_pos_bias is not None:
            bias = self.key_pos_bias[:position].to(keys.dtype)
            keys = keys + bias[:, None, None, :]
        query = self.w_query[position]
        scale = math.sqrt(values.shape[-1]) * self.base_temperature
        scores = torch.einsum("h,nbth->nbt", query.float(), keys.float()) / scale
        alpha = torch.softmax(scores, dim=0)
        routed = torch.einsum("nbt,nbth->bth", alpha, values.float()).to(values.dtype)
        return routed, alpha.permute(1, 2, 0)  # α as [B, T, n]


class Qwen3VLRetrofit(nn.Module):
    """Wrap a pretrained Qwen3VLForConditionalGeneration with block-level
    AttnRes routing between groups of text decoder layers."""

    def __init__(
        self,
        base_model,
        num_blocks: int = 14,
        route: str = "A",
        beta_init_logit: float = -2.2,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        assert route in ("A", "B"), route
        self.base_model = base_model
        self.route = route

        cfg = base_model.config.text_config
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_hidden_layers
        assert self.num_layers % num_blocks == 0, (
            f"num_hidden_layers={self.num_layers} must divide num_blocks={num_blocks}"
        )
        self.num_blocks = num_blocks
        self.layers_per_block = self.num_layers // num_blocks

        self.router = BlockAttnResQwen(
            hidden_size=self.hidden_size,
            num_blocks=num_blocks + 1,  # +1 for embedding source
            use_positional_bias=True,
            initializer_range=initializer_range,
        )

        if route == "A":
            self.gate_logits = nn.Parameter(
                torch.full((num_blocks,), beta_init_logit)
            )
        else:
            self.register_parameter("gate_logits", None)

    # ── Accessors ──────────────────────────────────────────────────

    @property
    def text_layers(self):
        return self.base_model.model.language_model.layers

    @property
    def text_embedding(self):
        # Qwen3VLModel has `embed_tokens` under language_model
        return self.base_model.model.language_model.embed_tokens

    @property
    def text_norm(self):
        return self.base_model.model.language_model.norm

    @property
    def lm_head(self):
        return self.base_model.lm_head

    # ── Base-LM forward (text only, no vision) ─────────────────────

    def _layer_forward(self, layer_idx, hidden, position_embeddings, attention_mask=None, **kwargs):
        """Run a single Qwen3VLTextDecoderLayer."""
        layer = self.text_layers[layer_idx]
        out = layer(
            hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Qwen text decoder layer returns (hidden,) or tuple
        if isinstance(out, tuple):
            return out[0]
        return out

    def _block_forward(self, block_idx, hidden, position_embeddings, attention_mask=None, **kwargs):
        start = block_idx * self.layers_per_block
        end = start + self.layers_per_block
        for l in range(start, end):
            hidden = self._layer_forward(l, hidden, position_embeddings, attention_mask, **kwargs)
        return hidden

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_alpha: bool = False,
        **kwargs,
    ) -> Qwen3VLRetrofitOutput:
        embeds = self.text_embedding(input_ids)

        # Prepare RoPE position embeddings
        device = embeds.device
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        # Qwen3VL uses mrope with 3 dimensions (text, spatial h, spatial w).
        # For text-only: use position_ids broadcast.
        if position_ids.dim() == 2:
            # expand to [3, B, T] for mrope (all same for text)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        rotary_emb = self.base_model.model.language_model.rotary_emb
        position_embeddings = rotary_emb(embeds, position_ids)

        completed: list[torch.Tensor] = [embeds]
        prev_block = embeds
        alpha_list = []
        entropy_accum = None

        for n in range(self.num_blocks):
            if self.route == "A":
                beta_n = torch.sigmoid(self.gate_logits[n])
                if n == 0:
                    block_input = embeds
                else:
                    routed, alpha = self.router.route(n + 1, completed)
                    block_input = (1 - beta_n) * prev_block + beta_n * routed
                    ent = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    entropy_accum = ent if entropy_accum is None else entropy_accum + ent
                    if return_alpha:
                        alpha_list.append(alpha)
                block_out = self._block_forward(
                    n, block_input, position_embeddings, attention_mask, **kwargs
                )
                prev_block = block_out
                completed.append(block_out)
            else:
                raise NotImplementedError("Route B not yet ported to Qwen3-VL")

        hidden_states = self.text_norm(completed[-1])
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shifted = torch.cat(
                [labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1
            )
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), shifted.view(-1), ignore_index=-100,
            )

        return Qwen3VLRetrofitOutput(
            last_hidden_state=hidden_states,
            loss=loss,
            entropy_penalty=entropy_accum,
            alpha_list=alpha_list if return_alpha else None,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def freeze_base(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def retrofit_parameters(self):
        """Only the new retrofit-added parameters (pseudo-queries, gate)."""
        result = list(self.router.parameters())
        if self.gate_logits is not None:
            result.append(self.gate_logits)
        return result
