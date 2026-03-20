"""
Core Attention Residual (AttnRes) mechanism.

Instead of fixed residual connections x_{l} = x_{l-1} + f_l(x_{l-1}),
AttnRes learns input-dependent weighted combinations over ALL previous
layer outputs using a softmax attention over the depth dimension:

    x_l = sum_{i=0}^{l-1} alpha_{i->l} * h_i

where alpha_{i->l} = softmax(w_l^T * h_i / sqrt(d)) over i in {0,...,l-1}

This provides:
1. Learned, input-dependent routing signals
2. Natural importance scores for layer skipping
3. A mechanism for adaptive computation depth
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerAttnRes(nn.Module):
    """Per-layer Attention Residual module.

    Each layer l has a learnable pseudo-query w_l that attends over
    the outputs of all previous layers (depth-wise KV cache).

    This replaces the standard residual connection with a learned
    weighted combination of all previous layer outputs.
    """

    def __init__(self, d_model: int, layer_idx: int, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.temperature = temperature

        # Learnable pseudo-query for this layer (decoupled from forward pass)
        self.w_query = nn.Parameter(torch.randn(d_model) * 0.02)

        # Optional key projection for source layers
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        layer_outputs: list[torch.Tensor],  # [h_0, h_1, ..., h_{l-1}], each (B, S, D)
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention-weighted combination of all previous layer outputs.

        Args:
            layer_outputs: List of tensors from layers 0..l-1, each (B, S, D)
            return_weights: If True, return the attention weights alpha_{i->l}

        Returns:
            combined: (B, S, D) weighted combination
            weights: (B, S, num_sources) attention weights if requested
        """
        num_sources = len(layer_outputs)
        if num_sources == 0:
            raise ValueError("Need at least one source layer output")

        if num_sources == 1:
            # Only embedding layer available - pass through
            weights = torch.ones(
                layer_outputs[0].shape[0], layer_outputs[0].shape[1], 1,
                device=layer_outputs[0].device, dtype=layer_outputs[0].dtype
            )
            if return_weights:
                return layer_outputs[0], weights
            return layer_outputs[0], None

        # Stack source outputs: (B, S, num_sources, D)
        sources = torch.stack(layer_outputs, dim=2)
        B, S, N, D = sources.shape

        # Project sources to keys: (B, S, num_sources, D)
        keys = self.key_proj(sources)

        # Compute attention scores: w_l^T * k_i / sqrt(d)
        # w_query: (D,) -> (1, 1, 1, D)
        query = self.w_query.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Scores: (B, S, num_sources)
        scores = (query * keys).sum(dim=-1) / (math.sqrt(D) * self.temperature)

        # Softmax over source layers (depth dimension)
        weights = F.softmax(scores, dim=-1)

        # Weighted combination: (B, S, D)
        combined = (weights.unsqueeze(-1) * sources).sum(dim=2)
        combined = self.norm(combined)

        if return_weights:
            return combined, weights
        return combined, None


class BlockAttnRes(nn.Module):
    """Block-level Attention Residual module.

    Groups L layers into N blocks and applies AttnRes at the block level.
    This provides coarser-grained routing that's more suitable for
    skip/loop decisions.

    Each block output attends over all previous block outputs to form
    its input, creating a natural routing granularity.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        block_idx: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.block_idx = block_idx

        # Per-block pseudo-query
        self.w_query = nn.Parameter(torch.randn(d_model) * 0.02)

        # Key projection
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

        # Value projection (optional, for richer representations)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.temperature = temperature

    def forward(
        self,
        block_outputs: list[torch.Tensor],  # outputs from blocks 0..n-1
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute block-level attention residual."""
        num_sources = len(block_outputs)

        if num_sources == 1:
            weights = torch.ones(
                block_outputs[0].shape[0], block_outputs[0].shape[1], 1,
                device=block_outputs[0].device, dtype=block_outputs[0].dtype
            )
            if return_weights:
                return block_outputs[0], weights
            return block_outputs[0], None

        # Stack: (B, S, N, D)
        sources = torch.stack(block_outputs, dim=2)
        B, S, N, D = sources.shape

        # Keys and values
        keys = self.key_proj(sources)
        values = self.value_proj(sources)

        # Attention scores
        query = self.w_query.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = (query * keys).sum(dim=-1) / (math.sqrt(D) * self.temperature)
        weights = F.softmax(scores, dim=-1)

        # Weighted combination
        combined = (weights.unsqueeze(-1) * values).sum(dim=2)
        combined = self.norm(combined)

        if return_weights:
            return combined, weights
        return combined, None


class OnlineSoftmaxMerge(nn.Module):
    """Online softmax merge for efficient AttnRes computation.

    Implements Algorithm 1 from the AttnRes paper: incrementally
    updates the weighted combination as new layer outputs arrive,
    without needing to store and re-attend over all previous outputs.

    This is crucial for:
    1. Memory efficiency during training
    2. Supporting dynamic skipping (just skip the update step)
    3. Enabling streaming/online inference
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        running_sum: torch.Tensor,      # Current weighted sum (B, S, D)
        running_max: torch.Tensor,       # Current max score (B, S)
        running_exp_sum: torch.Tensor,   # Current sum of exp(scores) (B, S)
        new_output: torch.Tensor,        # New layer output (B, S, D)
        new_score: torch.Tensor,         # Score for new layer (B, S)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Incrementally update the attention-weighted combination.

        Uses the online softmax trick to maintain numerical stability
        while processing one layer at a time.
        """
        # New maximum
        new_max = torch.maximum(running_max, new_score)

        # Rescale old sum
        old_scale = torch.exp(running_max - new_max)
        new_scale = torch.exp(new_score - new_max)

        # Update exponential sum
        new_exp_sum = running_exp_sum * old_scale + new_scale

        # Update weighted sum
        new_sum = (
            running_sum * old_scale.unsqueeze(-1) +
            new_output * new_scale.unsqueeze(-1)
        )

        return new_sum, new_max, new_exp_sum

    def finalize(
        self,
        running_sum: torch.Tensor,
        running_exp_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the running sum to get final output."""
        return running_sum / running_exp_sum.unsqueeze(-1).clamp(min=1e-8)
