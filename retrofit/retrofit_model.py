"""Retrofit a pretrained standard transformer with AttnRes routing.

Takes a `TransformerForCausalLM` (from fla.models.transformer) and wraps
its layers into an AttnRes-capable model supporting three retrofit routes:

  Route A — Interpolation gate:
    block_n_input = (1 - β_n) · prev_block_output + β_n · AttnRes(all_prev)
    β starts at 0 so initial behavior equals the original model.

  Route B — Auxiliary observer (distillation):
    Forward unchanged (standard residual chain).
    AttnRes pseudo-queries trained via a distillation loss to match
    prev_block_output as a weighted combination of earlier outputs.
    Skip decisions use the learned α at inference.

  Route C — Temperature annealing + informed init:
    block_n_input = softmax(w_n · k_i / (τ·√d)) · block_i_outputs
    w_n initialized such that α peaks at i=n-1 (near-identity).
    τ anneals from low (sharp, preserves original) to 1 during fine-tune.

All routes operate at block granularity: the original N_L layers are
grouped into N_B blocks (N_L / N_B layers per block). AttnRes attends
over block boundaries, not individual layers.

Only the pseudo-queries, key projection, optional gate β, and optional
temperature logits are added. The original transformer weights are
preserved and can be frozen or fine-tuned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Core routing module
# ──────────────────────────────────────────────────────────────────────


class BlockAttnRes(nn.Module):
    """Per-block-position routing over previously completed block outputs.

    Each block position n has its own pseudo-query w_n.  At position n we
    compute α_{i→n} = softmax_{i<n}(w_n · k_i / (τ·√d)) and combine the
    previous block outputs as ∑_i α_{i→n} h_i.

    Parameters
    ----------
    hidden_size: int
    num_blocks: int  (N_B — number of block positions including the embedding)
    temperature: float  (base τ; multiplied by τ_learnable if `learnable_tau=True`)
    learnable_tau: bool
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        temperature: float = 1.0,
        learnable_tau: bool = False,
        use_positional_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.base_temperature = temperature
        self.use_positional_bias = use_positional_bias

        # One pseudo-query per block position (excluding position 0 which is
        # the embedding; we compute routing ONLY when entering positions 1..N-1).
        self.w_query = nn.Parameter(torch.empty(num_blocks, hidden_size))
        # Key projection (shared across sources)
        self.key_norm = nn.RMSNorm(hidden_size, eps=1e-6)

        # Positional bias on keys: makes each source position distinguishable
        # even when block outputs are similar (critical for retrofit on
        # pretrained standard transformers where consecutive layer outputs
        # are architecturally close in representation space).
        if use_positional_bias:
            self.key_pos_bias = nn.Parameter(torch.empty(num_blocks, hidden_size))
        else:
            self.register_parameter("key_pos_bias", None)

        if learnable_tau:
            # Per-position temperature multiplier, init to 0 → τ_eff = τ_base
            self.log_tau_multiplier = nn.Parameter(torch.zeros(num_blocks))
        else:
            self.register_parameter("log_tau_multiplier", None)

    def reset_parameters(self, initializer_range: float = 0.02):
        nn.init.normal_(self.w_query, mean=0.0, std=initializer_range)
        if hasattr(self.key_norm, "weight") and self.key_norm.weight is not None:
            nn.init.ones_(self.key_norm.weight)
        if self.log_tau_multiplier is not None:
            nn.init.zeros_(self.log_tau_multiplier)
        if self.key_pos_bias is not None:
            nn.init.normal_(self.key_pos_bias, mean=0.0, std=initializer_range)

    def set_informed_init(self, block_outputs: list[torch.Tensor], scale: float = 1.0):
        """Init w_n = scale · normalize( mean( k_{n-1} ) ) over calibration data.

        The mean of unit-ish keys is not unit-normed (it shrinks), so we
        renormalize to restore a sensible dot-product magnitude, and
        optionally scale up further to sharpen the softmax.

        `block_outputs[i]` is the aggregated key for block i (shape [hidden_size]).
        """
        with torch.no_grad():
            for n in range(1, self.num_blocks):
                if n - 1 < len(block_outputs):
                    k_mean = block_outputs[n - 1].float()
                    k_mean = k_mean / (k_mean.norm() + 1e-6)
                    # Restore magnitude consistent with RMS-normed keys which
                    # have ≈ √d norm, so dot-product ≈ √d = 32 for d=1024.
                    k_mean = k_mean * math.sqrt(self.hidden_size) * scale
                    self.w_query[n].copy_(k_mean.to(self.w_query.dtype))
            # Keep w_0 random (never actually used for routing)

    def effective_temperature(self, position: int | None = None) -> torch.Tensor:
        """Returns a scalar tensor for the effective temperature at `position`
        (or a vector of all positions if position is None)."""
        base = torch.tensor(self.base_temperature, device=self.w_query.device)
        if self.log_tau_multiplier is None:
            return base
        if position is None:
            return base * self.log_tau_multiplier.exp()
        return base * self.log_tau_multiplier[position].exp()

    def route(
        self,
        position: int,
        completed_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute AttnRes output at block position `position`.

        Args:
            position: Current block position n (>=1).
            completed_outputs: List of block outputs [h_0, ..., h_{n-1}],
                each of shape [batch, seq, hidden]. h_0 is the embedding
                output (input to block 1).

        Returns:
            routed_output: [batch, seq, hidden] — weighted combination.
            alpha: [batch, seq, n] — routing weights over sources.
        """
        assert position >= 1
        assert len(completed_outputs) == position
        # Stack sources into [n, batch, seq, hidden]
        values = torch.stack(completed_outputs, dim=0)  # [n, B, T, H]
        # Compute keys via RMSNorm (no projection matrix — just normalized source)
        keys = self._rms_norm(values)  # [n, B, T, H]
        # Add positional bias so each source position is distinguishable
        # even when content is similar. bias[i] is broadcast across batch/seq.
        if self.key_pos_bias is not None:
            bias = self.key_pos_bias[:position].to(keys.dtype)  # [n, H]
            keys = keys + bias[:, None, None, :]
        query = self.w_query[position]  # [H]
        tau = self.effective_temperature(position)
        scale = math.sqrt(values.shape[-1]) * tau
        # scores[i, B, T] = <query, keys[i, B, T]> / scale
        scores = torch.einsum("h,nbth->nbt", query.float(), keys.float()) / scale  # [n, B, T]
        # softmax over source dim
        alpha = torch.softmax(scores, dim=0)  # [n, B, T]
        # combine in fp32 for numerical stability, cast back
        routed = torch.einsum("nbt,nbth->bth", alpha, values.float()).to(values.dtype)
        # Transpose α to [B, T, n]
        alpha_bth = alpha.permute(1, 2, 0)
        return routed, alpha_bth

    @staticmethod
    def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        fp = x.float()
        inv = torch.rsqrt(fp.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (fp * inv).to(x.dtype)


# ──────────────────────────────────────────────────────────────────────
# Retrofit wrapper
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RetrofitOutput:
    last_hidden_state: torch.Tensor
    loss: torch.Tensor | None = None
    distill_loss: torch.Tensor | None = None
    entropy_penalty: torch.Tensor | None = None  # NEGATIVE α entropy (to MINIMIZE → sparser α)
    alpha_list: list[torch.Tensor] | None = None


class RetrofitModel(nn.Module):
    """Wraps a TransformerForCausalLM, grouping its layers into N blocks and
    adding AttnRes routing between blocks.

    Parameters
    ----------
    base_model: TransformerForCausalLM
        Pretrained standard transformer.
    num_blocks: int
        Number of block positions (must divide the original num_hidden_layers).
        e.g. for 12 layers with num_blocks=6, each block = 2 consecutive layers.
    route: "A" | "B" | "C"
        Retrofit strategy.
    """

    ROUTES = ("A", "B", "C")

    def __init__(
        self,
        base_model: nn.Module,
        num_blocks: int = 6,
        route: str = "A",
        initializer_range: float = 0.02,
    ):
        super().__init__()
        assert route in self.ROUTES, f"route must be in {self.ROUTES}"
        self.base_model = base_model
        self.route = route

        config = base_model.config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        assert self.num_layers % num_blocks == 0, (
            f"num_hidden_layers={self.num_layers} must be divisible by "
            f"num_blocks={num_blocks}"
        )
        self.num_blocks = num_blocks
        self.layers_per_block = self.num_layers // num_blocks

        # Block-level AttnRes router
        # We have (num_blocks + 1) "source positions": embedding (index 0) plus
        # each block's output (indices 1..num_blocks). Pseudo-queries are at
        # positions 1..num_blocks (one per block's INPUT).
        self.router = BlockAttnRes(
            hidden_size=self.hidden_size,
            num_blocks=num_blocks + 1,  # index 0 is embedding; indices 1..N map to blocks
            learnable_tau=(route == "C"),
        )
        self.router.reset_parameters(initializer_range)

        # Route A: per-block interpolation gate β
        if route == "A":
            # β_n ∈ (0, 1) via sigmoid. init logit determines cold-start tradeoff:
            #   0.0  → β=0.5 (50% AttnRes, strong gradient to pseudo-queries,
            #                  larger init quality hit but learning starts)
            #  -2.0  → β≈0.12 (moderate mix, slower but smoother)
            #  -6.0  → β≈0.0025 (essentially standard residual; β does not move)
            self.gate_logits = nn.Parameter(torch.full((num_blocks,), -2.2))
        else:
            self.register_parameter("gate_logits", None)

    # ── Accessors ──────────────────────────────────────────────────

    @property
    def embedding(self):
        return self.base_model.model.embeddings

    @property
    def layers(self):
        return self.base_model.model.layers

    @property
    def final_norm(self):
        return self.base_model.model.norm

    @property
    def lm_head(self):
        return self.base_model.lm_head

    def block_forward(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        """Run layers belonging to block `block_idx` and return the output."""
        start = block_idx * self.layers_per_block
        end = start + self.layers_per_block
        for l in range(start, end):
            layer = self.layers[l]
            outs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                **kwargs,
            )
            hidden_states = outs[0]
        return hidden_states

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_alpha: bool = False,
        **kwargs,
    ) -> RetrofitOutput:
        embeds = self.embedding(input_ids)
        completed: list[torch.Tensor] = [embeds]
        alpha_list: list[torch.Tensor] = []
        distill_accum = None
        entropy_accum = None

        # For Route A we need to track prev_block_output (standard chain)
        prev_block_output = embeds

        for n in range(self.num_blocks):
            # Block n consumes input at source-position (n+1) of the router
            # i.e., it attends to completed sources [0..n] (embedding + first n blocks)
            if self.route == "A":
                # β_n: sigmoid(logit)
                beta_n = torch.sigmoid(self.gate_logits[n])
                if n == 0:
                    # No previous blocks to route over; input is just embedding
                    block_input = embeds
                else:
                    routed, alpha = self.router.route(
                        position=n + 1, completed_outputs=completed
                    )
                    block_input = (1 - beta_n) * prev_block_output + beta_n * routed
                    # Accumulate entropy for regularizer
                    ent = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    entropy_accum = ent if entropy_accum is None else entropy_accum + ent
                    if return_alpha:
                        alpha_list.append(alpha)
                block_out = self.block_forward(
                    n, block_input, attention_mask=attention_mask, **kwargs
                )
                prev_block_output = block_out
                completed.append(block_out)

            elif self.route == "B":
                # Forward is unchanged (standard residual chain)
                block_input = prev_block_output
                block_out = self.block_forward(
                    n, block_input, attention_mask=attention_mask, **kwargs
                )
                # Distillation: router should predict block_input from completed
                if n >= 1:
                    routed, alpha = self.router.route(
                        position=n + 1, completed_outputs=completed
                    )
                    diff = routed - block_input.detach()
                    distill = (diff * diff).mean()
                    distill_accum = (
                        distill if distill_accum is None else distill_accum + distill
                    )
                    # Accumulate α entropy for regularizer
                    ent = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    entropy_accum = ent if entropy_accum is None else entropy_accum + ent
                    if return_alpha:
                        alpha_list.append(alpha)
                prev_block_output = block_out
                completed.append(block_out)

            elif self.route == "C":
                # Pure AttnRes (routed input to block)
                if n == 0:
                    block_input = embeds
                else:
                    routed, alpha = self.router.route(
                        position=n + 1, completed_outputs=completed
                    )
                    block_input = routed
                    if return_alpha:
                        alpha_list.append(alpha)
                block_out = self.block_forward(
                    n, block_input, attention_mask=attention_mask, **kwargs
                )
                prev_block_output = block_out
                completed.append(block_out)
            else:
                raise ValueError(self.route)

        # Final norm + lm_head
        hidden_states = self.final_norm(completed[-1])
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Standard causal LM shift
            shifted = torch.cat(
                [labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1
            )
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), shifted.view(-1),
                ignore_index=-100,
            )

        return RetrofitOutput(
            last_hidden_state=hidden_states,
            loss=loss,
            distill_loss=distill_accum,
            entropy_penalty=entropy_accum,
            alpha_list=alpha_list if return_alpha else None,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def freeze_base(self):
        """Freeze the pretrained transformer weights (only retrofit params train)."""
        for p in self.base_model.parameters():
            p.requires_grad = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def collect_block_keys_for_init(
        self,
        calibration_batches: list[torch.LongTensor],
        device: str = "cuda",
    ) -> list[torch.Tensor]:
        """For Route C: compute mean key per block over calibration data.
        Returns list of [hidden_size] tensors, one per completed source.
        """
        self.eval()
        sum_by_idx: dict[int, torch.Tensor] = {}
        count_by_idx: dict[int, int] = {}
        for batch in calibration_batches:
            batch = batch.to(device)
            embeds = self.embedding(batch)
            completed = [embeds]
            prev = embeds
            for n in range(self.num_blocks):
                out = self.block_forward(n, prev, attention_mask=None)
                completed.append(out)
                prev = out
            # Aggregate keys (normalized)
            for idx, h in enumerate(completed):
                k = BlockAttnRes._rms_norm(h)
                # Average over batch and seq
                mean_k = k.mean(dim=(0, 1)).to(torch.float32)
                if idx not in sum_by_idx:
                    sum_by_idx[idx] = mean_k
                    count_by_idx[idx] = 1
                else:
                    sum_by_idx[idx] += mean_k
                    count_by_idx[idx] += 1
        result = []
        for idx in sorted(sum_by_idx.keys()):
            result.append(sum_by_idx[idx] / count_by_idx[idx])
        return result
