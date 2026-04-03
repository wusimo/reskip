"""
AttnRes-Routed Block Looping Transformer.

Instead of L unique blocks, maintains K < L shared-weight blocks
and lets AttnRes decide how many times to loop through them.

This is a learned, continuous relaxation of Universal Transformer-style
looping, where AttnRes naturally differentiates the role of each
application of the same block through depth-position-specific pseudo-queries.

Key advantage over naive Universal Transformers:
- The pseudo-query w_l is per-position-in-depth, NOT per-block
- Even when blocks share weights, AttnRes routing is unique per depth position
- The model can learn "apply block 2 three times for math, once for factual recall"
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_residual import BlockAttnRes
from .adaptive_transformer import (
    AdaptiveTransformerConfig,
    TransformerBlock,
)


@dataclass
class LoopingTransformerConfig(AdaptiveTransformerConfig):
    """Config for looping transformer."""
    n_unique_blocks: int = 4       # K: number of unique weight-sharing groups
    max_loop_depth: int = 12       # K * max_loops: total depth positions
    loop_warmup_steps: int = 1000  # Steps before introducing weight sharing

    # Halting mechanism
    use_adaptive_halting: bool = True
    halt_threshold: float = 0.95   # Cumulative confidence threshold for halting
    halt_penalty_weight: float = 0.01  # Ponder cost regularization


class LoopingTransformerWithAttnRes(nn.Module):
    """
    Transformer with shared-weight blocks and AttnRes-guided looping.

    Architecture:
    - K unique transformer blocks (weight-sharing groups)
    - Each block can be applied up to max_loops times
    - AttnRes pseudo-queries are unique per depth position (not per block)
    - This allows the model to learn different roles for each application
      of the same block

    The depth-wise KV cache for AttnRes includes outputs from ALL
    applications of ALL blocks, enabling rich routing patterns.

    Training curriculum:
    1. Start with K*M unique blocks (no weight sharing)
    2. Progressively merge weights within groups
    3. Fine-tune with full weight sharing
    """

    def __init__(self, config: LoopingTransformerConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = None if config.use_rope else nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # K unique blocks
        layers_per_block = max(1, config.n_layers // config.n_unique_blocks)
        self.unique_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(config) for _ in range(layers_per_block)
            ])
            for _ in range(config.n_unique_blocks)
        ])

        # AttnRes for each depth position (K * max_loops positions)
        self.block_attn_res = nn.ModuleList([
            BlockAttnRes(
                config.d_model, config.max_loop_depth, i,
                temperature=config.attn_res_temperature,
            )
            for i in range(config.max_loop_depth)
        ])

        # Adaptive halting: per-position halting probability
        if config.use_adaptive_halting:
            self.halt_proj = nn.Linear(config.d_model, 1)

        # Output
        self.out_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_block_schedule(self) -> list[int]:
        """
        Get the block application schedule.

        Default: cycle through blocks [0, 1, ..., K-1, 0, 1, ..., K-1, ...]
        up to max_loop_depth positions.
        """
        schedule = []
        for pos in range(self.config.max_loop_depth):
            block_idx = pos % self.config.n_unique_blocks
            schedule.append(block_idx)
        return schedule

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
    ) -> dict:
        """
        Forward pass with adaptive looping.

        The model cycles through K unique blocks, with AttnRes deciding
        whether each additional application is useful. When adaptive
        halting is enabled, computation stops when cumulative halting
        probability exceeds the threshold.
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        if self.pos_emb is not None:
            positions = torch.arange(S, device=device).unsqueeze(0)
            x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))
        else:
            x = self.emb_dropout(self.token_emb(input_ids))

        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(S, S, device=device)).unsqueeze(0).unsqueeze(0)

        block_outputs = [x]
        routing_weights = []
        halt_probs = []
        block_schedule = self.get_block_schedule()

        # Running halting state
        cumulative_halt = torch.zeros(B, S, device=device)
        remainders = torch.zeros(B, S, device=device)
        n_updates = torch.zeros(B, S, device=device)

        # Accumulated output (ACT-style)
        accumulated_output = torch.zeros(B, S, self.config.d_model, device=device)

        for depth_pos, block_idx in enumerate(block_schedule):
            if depth_pos == 0:
                continue  # First position is the embedding

            if depth_pos >= len(self.block_attn_res):
                break

            # AttnRes input
            attn_res_input, weights = self.block_attn_res[depth_pos](
                block_outputs, return_weights=True
            )
            routing_weights.append(weights)

            # Execute block
            h = attn_res_input
            for layer in self.unique_blocks[block_idx]:
                h = layer(h, mask)

            block_outputs.append(h)

            # Adaptive halting
            if self.config.use_adaptive_halting:
                halt_logit = self.halt_proj(h).squeeze(-1)  # (B, S)
                halt_prob = torch.sigmoid(halt_logit)
                halt_probs.append(halt_prob)

                # Determine which positions are still computing
                still_running = (cumulative_halt < 1.0).float()

                # Update cumulative halting probability
                new_halt = cumulative_halt + halt_prob * still_running

                # Positions that halt at this step
                halted_now = (new_halt >= self.config.halt_threshold).float() * still_running

                # Compute remainder for halted positions
                remainder = 1.0 - cumulative_halt

                # Update accumulated output
                weight = halt_prob * still_running * (1.0 - halted_now) + remainder * halted_now
                accumulated_output = accumulated_output + weight.unsqueeze(-1) * h

                # Update state
                cumulative_halt = new_halt
                n_updates = n_updates + still_running

                # Check if all positions have halted (inference only)
                if not self.training and (cumulative_halt >= self.config.halt_threshold).all():
                    break
            else:
                accumulated_output = h  # Just use last output

        # Final output
        if self.config.use_adaptive_halting:
            output = self.out_norm(accumulated_output)
        else:
            output = self.out_norm(block_outputs[-1])

        logits = self.lm_head(output)

        result = {"logits": logits}

        if return_routing_info:
            result["routing_weights"] = routing_weights
            result["halt_probs"] = halt_probs
            result["n_updates"] = n_updates
            result["effective_depth"] = n_updates.mean().item()

        return result

    def compute_ponder_cost(self, n_updates: torch.Tensor) -> torch.Tensor:
        """
        Compute ponder cost regularization (from ACT paper).

        Encourages the model to halt as early as possible while
        maintaining accuracy.
        """
        return self.config.halt_penalty_weight * n_updates.mean()

    def forward_with_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Forward pass with language modeling loss and ponder cost."""
        result = self.forward(input_ids, return_routing_info=True)

        # LM loss
        shift_logits = result["logits"][:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        result["lm_loss"] = lm_loss

        # Ponder cost
        if self.config.use_adaptive_halting and "n_updates" in result:
            ponder_cost = self.compute_ponder_cost(result["n_updates"])
            result["ponder_cost"] = ponder_cost
            result["loss"] = lm_loss + ponder_cost
        else:
            result["loss"] = lm_loss

        return result
