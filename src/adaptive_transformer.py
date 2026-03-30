"""
Adaptive Transformer with AttnRes-guided layer skipping and block looping.

This module implements the core architecture that uses Attention Residuals
as routing signals for:
1. Dynamic layer/block skipping based on importance thresholds
2. Adaptive block looping (weight-shared blocks applied variable times)
3. Input-dependent effective depth

Key insight: The AttnRes pseudo-queries are decoupled from the forward
computation, allowing routing decisions BEFORE executing layers.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_residual import BlockAttnRes, OnlineSoftmaxMerge


@dataclass
class AdaptiveTransformerConfig:
    """Configuration for adaptive transformer."""
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    n_blocks: int = 4           # Number of unique blocks (for looping)
    max_loops: int = 3          # Max times a block can be applied
    d_ff: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.1
    use_rope: bool = False
    rope_base: float = 10000.0

    # AttnRes parameters
    attn_res_temperature: float = 1.0
    use_attn_res: bool = True

    # Skipping parameters
    skip_threshold: float = 0.01  # epsilon for importance-based skipping
    enable_skipping: bool = False  # Enable at inference time

    # Looping parameters
    enable_looping: bool = False
    loop_threshold: float = 0.05  # Min attention weight to justify another loop


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, rope_base)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, HD)
        q, k, v = qkv.unbind(0)

        if self.use_rope:
            cos, sin = self.rotary(q, S)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Precompute rotary frequencies for attention heads."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=x.dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=x.dtype)
        return cos, sin


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, config: AdaptiveTransformerConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
        )
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.attn_norm(x), mask)
        # Pre-norm FFN with residual
        x = x + self.ff(self.ff_norm(x))
        return x


class AdaptiveTransformerWithAttnRes(nn.Module):
    """
    Transformer with Block AttnRes for adaptive computation.

    Architecture:
    - L layers grouped into N blocks
    - Each block has a BlockAttnRes module that attends over all
      previous block outputs
    - At inference, blocks can be skipped based on AttnRes importance
    - Optionally, blocks share weights and can be looped

    Training mode: All blocks execute, AttnRes weights are learned
    Inference mode: Blocks are skipped/looped based on AttnRes routing
    """

    def __init__(self, config: AdaptiveTransformerConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = None if config.use_rope else nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Determine block structure
        layers_per_block = config.n_layers // config.n_blocks
        self.layers_per_block = layers_per_block

        if config.enable_looping:
            # Shared-weight blocks (K unique blocks, each loopable)
            self.blocks = nn.ModuleList([
                nn.ModuleList([
                    TransformerBlock(config) for _ in range(layers_per_block)
                ])
                for _ in range(config.n_blocks)
            ])
        else:
            # Unique blocks
            self.blocks = nn.ModuleList([
                nn.ModuleList([
                    TransformerBlock(config) for _ in range(layers_per_block)
                ])
                for _ in range(config.n_layers // layers_per_block)
            ])

        # Block AttnRes modules - one per depth position
        # For looping: need max_depth = n_blocks * max_loops positions
        if config.enable_looping:
            max_depth = config.n_blocks * config.max_loops
        else:
            max_depth = len(self.blocks)

        self.block_attn_res = nn.ModuleList([
            BlockAttnRes(
                config.d_model, max_depth, i,
                temperature=config.attn_res_temperature,
            )
            for i in range(max_depth)
        ])

        # Online softmax merger for efficient computation
        self.online_merger = OnlineSoftmaxMerge(config.d_model)

        # Output head
        self.out_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.token_emb.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def compute_block_importance(
        self,
        block_outputs: list[torch.Tensor],
        depth_pos: int,
    ) -> torch.Tensor:
        """
        Compute importance score for a block at given depth position.

        I(n) = max_{l > n} alpha_{n->l}

        Approximated by checking the current depth position's AttnRes
        weights for the most recent block output.

        Returns: (B, S) importance scores
        """
        if depth_pos >= len(self.block_attn_res):
            return torch.ones(
                block_outputs[0].shape[0], block_outputs[0].shape[1],
                device=block_outputs[0].device
            )

        _, weights = self.block_attn_res[depth_pos](
            block_outputs, return_weights=True
        )
        # weights: (B, S, num_sources)
        # Return the weight for the most recent block output
        return weights[:, :, -1]  # (B, S)

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, S)
        mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
    ) -> dict:
        """
        Forward pass with AttnRes-guided adaptive computation.

        Returns:
            dict with:
                - logits: (B, S, V) output logits
                - routing_weights: list of (B, S, num_sources) attention weights
                - effective_depth: (B, S) average effective depth per token
                - blocks_executed: list of which blocks were actually executed
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

        # Track block outputs for AttnRes
        block_outputs = [x]  # h_0 = embedding
        routing_weights = []
        blocks_executed = []

        depth_pos = 0

        if self.config.enable_looping:
            # === Looping mode ===
            # Apply blocks with weight sharing, AttnRes decides looping
            for loop_iter in range(self.config.max_loops):
                for block_idx, block in enumerate(self.blocks):
                    depth_pos = loop_iter * self.config.n_blocks + block_idx + 1

                    if depth_pos >= len(self.block_attn_res):
                        break

                    # AttnRes: attend over all previous block outputs
                    attn_res_input, weights = self.block_attn_res[depth_pos](
                        block_outputs, return_weights=True
                    )
                    routing_weights.append(weights)

                    # Check if this loop iteration is worth executing
                    if (self.config.enable_skipping and
                        not self.training and
                        loop_iter > 0):
                        # For loops after the first pass, check importance
                        importance = weights[:, :, -1].mean()
                        if importance < self.config.loop_threshold:
                            continue

                    # Execute block layers
                    h = attn_res_input
                    for layer in block:
                        h = layer(h, mask)

                    block_outputs.append(h)
                    blocks_executed.append((block_idx, loop_iter))
        else:
            # === Standard mode (with optional skipping) ===
            for block_idx, block in enumerate(self.blocks):
                depth_pos = block_idx + 1

                if depth_pos >= len(self.block_attn_res):
                    break

                # AttnRes: attend over all previous block outputs
                attn_res_input, weights = self.block_attn_res[depth_pos](
                    block_outputs, return_weights=True
                )
                routing_weights.append(weights)

                # Check if this block should be skipped
                if (self.config.enable_skipping and
                    not self.training and
                    block_idx > 0 and block_idx < len(self.blocks) - 1):
                    # Never skip first or last block
                    importance = weights[:, :, -1].mean()
                    if importance < self.config.skip_threshold:
                        # Skip this block - don't add to block_outputs
                        # The next block will attend over existing outputs
                        blocks_executed.append((block_idx, -1))  # -1 = skipped
                        continue

                # Execute block layers
                h = attn_res_input
                for layer in block:
                    h = layer(h, mask)

                block_outputs.append(h)
                blocks_executed.append((block_idx, 0))

        # Final output from last block output
        output = self.out_norm(block_outputs[-1])
        logits = self.lm_head(output)

        result = {"logits": logits}

        if return_routing_info:
            result["routing_weights"] = routing_weights
            result["blocks_executed"] = blocks_executed
            result["num_blocks_executed"] = sum(
                1 for _, status in blocks_executed if status >= 0
            )
            result["block_outputs"] = block_outputs

        return result

    def get_routing_statistics(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Analyze routing patterns without skipping.

        Returns detailed statistics about which blocks the model
        considers important for given inputs.
        """
        # Force full execution
        old_skip = self.config.enable_skipping
        self.config.enable_skipping = False

        result = self.forward(input_ids, mask, return_routing_info=True)

        self.config.enable_skipping = old_skip

        # Compute per-block importance scores
        # I(n) = max_{l > n} alpha_{n->l}
        weights_list = result["routing_weights"]
        num_positions = len(weights_list)

        # Build importance matrix
        importance_matrix = torch.zeros(num_positions, num_positions)
        for l, weights in enumerate(weights_list):
            # weights: (B, S, num_sources) - average over batch and sequence
            avg_weights = weights.mean(dim=(0, 1))  # (num_sources,)
            for i, w in enumerate(avg_weights):
                importance_matrix[i, l] = w.item()

        # Per-block importance: max downstream attention
        block_importance = []
        for n in range(num_positions):
            if n < num_positions - 1:
                imp = importance_matrix[n, n+1:].max().item()
            else:
                imp = 1.0  # Last block is always important
            block_importance.append(imp)

        return {
            "importance_matrix": importance_matrix,
            "block_importance": block_importance,
            "routing_weights": weights_list,
            "effective_depth": result["num_blocks_executed"],
        }


class AdaptiveTransformerForCausalLM(nn.Module):
    """Wrapper with training utilities for causal language modeling."""

    def __init__(self, config: AdaptiveTransformerConfig):
        super().__init__()
        self.model = AdaptiveTransformerWithAttnRes(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
    ) -> dict:
        result = self.model(input_ids, return_routing_info=return_routing_info)

        if labels is not None:
            # Shift logits and labels for causal LM loss
            shift_logits = result["logits"][:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

            # Optional routing diversity loss to encourage non-degenerate routing
            if return_routing_info and "routing_weights" in result:
                routing_entropy = self._routing_entropy(result["routing_weights"])
                result["routing_entropy"] = routing_entropy

        return result

    def _routing_entropy(self, routing_weights: list[torch.Tensor]) -> torch.Tensor:
        """Compute average entropy of routing distributions.

        Higher entropy = more distributed attention = less skippable blocks.
        We track this as a diagnostic, not as a loss term (the model should
        learn to concentrate naturally).
        """
        total_entropy = 0.0
        count = 0
        for weights in routing_weights:
            # weights: (B, S, num_sources)
            # Entropy per position
            entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
            total_entropy += entropy
            count += 1
        return total_entropy / max(count, 1)


class StandardTransformerForCausalLM(nn.Module):
    """Baseline transformer without AttnRes routing."""

    def __init__(self, config: AdaptiveTransformerConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = None if config.use_rope else nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.out_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
    ) -> dict:
        del return_routing_info
        _, seq_len = input_ids.shape
        device = input_ids.device

        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))
        else:
            x = self.emb_dropout(self.token_emb(input_ids))

        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(self.out_norm(x))
        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return result
