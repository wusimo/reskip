"""
VLA-specific Adaptive Depth with AttnRes.

Extends the adaptive transformer for Vision-Language-Action models,
where different token modalities (vision, language, action) may
benefit from different effective depths.

Key insight: In a VLA, vision tokens encode perceptual features
(captured early), language tokens encode task specifications (moderate
depth), and action tokens require compositional motor planning
(deepest processing). AttnRes naturally discovers this structure.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_residual import BlockAttnRes
from .adaptive_transformer import (
    AdaptiveTransformerConfig,
    TransformerBlock,
    MultiHeadAttention,
    FeedForward,
)


class TokenModality(IntEnum):
    """Token modality types for per-modality routing analysis."""
    VISION = 0
    LANGUAGE = 1
    ACTION = 2


@dataclass
class VLAAdaptiveConfig(AdaptiveTransformerConfig):
    """Configuration for VLA-specific adaptive transformer."""
    # Vision encoder
    vision_dim: int = 1024
    vision_patch_size: int = 14
    vision_seq_len: int = 256      # Number of vision tokens

    # Action head
    action_dim: int = 7             # DoF (e.g., 6 + gripper)
    action_chunk_size: int = 16     # Number of future actions to predict

    # Modality-specific skipping thresholds
    vision_skip_threshold: float = 0.02   # More aggressive skipping
    language_skip_threshold: float = 0.01
    action_skip_threshold: float = 0.005  # Less aggressive - actions need depth

    # Whether to use modality-aware routing
    modality_aware_routing: bool = True


class VisionProjector(nn.Module):
    """Projects vision encoder outputs to transformer dimension."""

    def __init__(self, vision_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.proj(vision_features)


class ActionHead(nn.Module):
    """Parallel action chunk decoder using cross-attention to backbone."""

    def __init__(self, config: VLAAdaptiveConfig):
        super().__init__()
        self.config = config

        # Learnable action queries
        self.action_queries = nn.Parameter(
            torch.randn(config.action_chunk_size, config.d_model) * 0.02
        )

        # Cross-attention from action queries to backbone output
        self.cross_attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.cross_norm = nn.LayerNorm(config.d_model)

        # Self-attention among action queries (bidirectional)
        self.self_attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.self_norm = nn.LayerNorm(config.d_model)

        # FFN
        self.ff = FeedForward(config.d_model, config.d_ff)
        self.ff_norm = nn.LayerNorm(config.d_model)

        # Output projection to action space
        self.action_proj = nn.Linear(config.d_model, config.action_dim)

    def forward(
        self,
        backbone_output: torch.Tensor,  # (B, S, D)
    ) -> torch.Tensor:
        """
        Predict action chunks from backbone representations.

        Returns: (B, action_chunk_size, action_dim)
        """
        B = backbone_output.shape[0]

        # Expand action queries for batch
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)

        # Self-attention among queries (no causal mask - bidirectional)
        queries = queries + self.self_attn(self.self_norm(queries))

        # Cross-attention to backbone
        # For simplicity, concatenate queries with backbone for attention
        # In practice, use proper cross-attention
        combined = torch.cat([queries, backbone_output], dim=1)
        cross_out = self.cross_attn(self.cross_norm(combined))
        queries = queries + cross_out[:, :self.config.action_chunk_size]

        # FFN
        queries = queries + self.ff(self.ff_norm(queries))

        # Project to action space
        actions = self.action_proj(queries)

        return actions


class VLAAdaptiveTransformer(nn.Module):
    """
    Vision-Language-Action Transformer with AttnRes-guided adaptive depth.

    Architecture:
    - Vision encoder (frozen) -> projector -> vision tokens
    - Language tokenizer -> language tokens
    - [Vision tokens | Language tokens] -> Adaptive Transformer backbone
    - Action head: cross-attention to backbone output -> action chunks

    The AttnRes mechanism provides modality-aware routing:
    - Vision tokens may exit early (perceptual features captured in early blocks)
    - Language tokens use moderate depth
    - Action tokens use full depth (compositional motor planning)

    At inference, per-modality skipping thresholds enable different
    effective depths for each modality.
    """

    def __init__(self, config: VLAAdaptiveConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Vision projector
        self.vision_proj = VisionProjector(config.vision_dim, config.d_model)

        # Modality embeddings
        self.modality_emb = nn.Embedding(3, config.d_model)  # vision, language, action

        # Transformer blocks
        layers_per_block = config.n_layers // config.n_blocks
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(config) for _ in range(layers_per_block)
            ])
            for _ in range(config.n_blocks)
        ])

        # Block AttnRes
        self.block_attn_res = nn.ModuleList([
            BlockAttnRes(
                config.d_model, config.n_blocks, i,
                temperature=config.attn_res_temperature,
            )
            for i in range(config.n_blocks)
        ])

        # Action head
        self.action_head = ActionHead(config)

        # Output norm
        self.out_norm = nn.LayerNorm(config.d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_modality_threshold(self, modality: TokenModality) -> float:
        """Get skip threshold for a specific modality."""
        if modality == TokenModality.VISION:
            return self.config.vision_skip_threshold
        elif modality == TokenModality.LANGUAGE:
            return self.config.language_skip_threshold
        else:
            return self.config.action_skip_threshold

    def forward(
        self,
        input_ids: torch.Tensor,              # (B, S_lang)
        vision_features: torch.Tensor,          # (B, S_vis, vision_dim)
        modality_ids: Optional[torch.Tensor] = None,  # (B, S_total)
        labels: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,  # (B, chunk, action_dim)
        return_routing_info: bool = False,
    ) -> dict:
        """
        Forward pass with modality-aware adaptive depth.

        Args:
            input_ids: Language token IDs
            vision_features: Pre-extracted vision features
            modality_ids: Per-token modality labels (0=vision, 1=language, 2=action)
            labels: Language modeling targets (optional)
            target_actions: Ground truth actions for training (optional)
            return_routing_info: Whether to return detailed routing analysis
        """
        B, S_lang = input_ids.shape
        S_vis = vision_features.shape[1]
        device = input_ids.device

        # Project vision features
        vis_tokens = self.vision_proj(vision_features)  # (B, S_vis, D)

        # Language embeddings
        positions = torch.arange(S_lang, device=device).unsqueeze(0)
        lang_tokens = self.token_emb(input_ids) + self.pos_emb(positions)

        # Concatenate: [vision | language]
        S_total = S_vis + S_lang
        x = torch.cat([vis_tokens, lang_tokens], dim=1)  # (B, S_total, D)

        # Add modality embeddings
        if modality_ids is None:
            modality_ids = torch.cat([
                torch.zeros(B, S_vis, dtype=torch.long, device=device),
                torch.ones(B, S_lang, dtype=torch.long, device=device),
            ], dim=1)
        x = x + self.modality_emb(modality_ids)
        x = self.emb_dropout(x)

        # Causal mask for full sequence
        mask = torch.tril(torch.ones(S_total, S_total, device=device))
        # Vision tokens can attend to all vision tokens (bidirectional within vision)
        mask[:S_vis, :S_vis] = 1.0
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Block-level forward with AttnRes
        block_outputs = [x]  # h_0 = embeddings
        routing_weights = []
        per_modality_weights = {m: [] for m in TokenModality}
        blocks_executed = []

        for block_idx, block in enumerate(self.blocks):
            depth_pos = block_idx + 1
            if depth_pos >= len(self.block_attn_res):
                break

            # AttnRes: attend over previous block outputs
            attn_res_input, weights = self.block_attn_res[depth_pos](
                block_outputs, return_weights=True
            )
            routing_weights.append(weights)

            # Per-modality weight analysis
            if return_routing_info:
                for mod in TokenModality:
                    mod_mask = (modality_ids == mod.value)  # (B, S_total)
                    if mod_mask.any():
                        # Average weights for this modality's tokens
                        mod_weights = weights[mod_mask]  # (num_mod_tokens, num_sources)
                        per_modality_weights[mod].append(mod_weights.mean(dim=0))

            # Modality-aware skipping at inference
            should_skip = False
            if (self.config.enable_skipping and
                not self.training and
                self.config.modality_aware_routing and
                block_idx > 0 and block_idx < len(self.blocks) - 1):

                # Check if ALL modalities want to skip this block
                skip_decisions = []
                for mod in TokenModality:
                    mod_mask = (modality_ids == mod.value)
                    if mod_mask.any():
                        mod_importance = weights[mod_mask][:, -1].mean()
                        threshold = self._get_modality_threshold(mod)
                        skip_decisions.append(mod_importance < threshold)

                should_skip = all(skip_decisions) if skip_decisions else False

            if should_skip:
                blocks_executed.append((block_idx, -1))
                continue

            # Execute block
            h = attn_res_input
            for layer in block:
                h = layer(h, mask)

            block_outputs.append(h)
            blocks_executed.append((block_idx, 0))

        # Final backbone output
        backbone_output = self.out_norm(block_outputs[-1])

        # Action prediction
        actions = self.action_head(backbone_output)

        result = {"actions": actions, "backbone_output": backbone_output}

        # Losses
        if target_actions is not None:
            action_loss = F.l1_loss(actions, target_actions)
            result["action_loss"] = action_loss
            result["loss"] = action_loss

        if return_routing_info:
            result["routing_weights"] = routing_weights
            result["per_modality_weights"] = per_modality_weights
            result["blocks_executed"] = blocks_executed

        return result

    def analyze_modality_depth(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Analyze per-modality depth utilization.

        Returns statistics on how much each modality attends to
        each block, revealing the effective depth per modality.
        """
        old_skip = self.config.enable_skipping
        self.config.enable_skipping = False

        result = self.forward(
            input_ids, vision_features, modality_ids,
            return_routing_info=True,
        )

        self.config.enable_skipping = old_skip

        analysis = {}
        for mod in TokenModality:
            mod_weights = result["per_modality_weights"][mod]
            if mod_weights:
                # Pad weights to same length (each block has different num_sources)
                max_len = max(w.shape[-1] for w in mod_weights)
                padded = []
                for w in mod_weights:
                    if w.shape[-1] < max_len:
                        pad = torch.zeros(max_len - w.shape[-1], device=w.device)
                        padded.append(torch.cat([w, pad]))
                    else:
                        padded.append(w)
                stacked = torch.stack(padded)
                analysis[mod.name] = {
                    "mean_weights_per_block": stacked.detach().cpu(),
                    "effective_depth": (stacked[:, -1] > 0.01).sum().item(),
                    "concentration": stacked.max(dim=-1).values.mean().item(),
                }

        return analysis
