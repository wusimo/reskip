"""Unit tests for all ReSkip modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from src.attn_residual import LayerAttnRes, BlockAttnRes, OnlineSoftmaxMerge
from src.adaptive_transformer import (
    AdaptiveTransformerConfig,
    AdaptiveTransformerWithAttnRes,
    AdaptiveTransformerForCausalLM,
)
from src.looping_transformer import (
    LoopingTransformerConfig,
    LoopingTransformerWithAttnRes,
)
from src.vla_adaptive import VLAAdaptiveConfig, VLAAdaptiveTransformer
from src.data import StructuredSyntheticLM, StructuredVLADataset, create_lm_dataloaders
from src.utils import count_transformer_flops, count_effective_flops, CosineWarmupScheduler


B, S, D = 2, 16, 64  # Small dims for testing


# ---- Core AttnRes ----

class TestLayerAttnRes:
    def test_single_source(self):
        module = LayerAttnRes(D, layer_idx=1)
        h0 = torch.randn(B, S, D)
        out, weights = module([h0], return_weights=True)
        assert out.shape == (B, S, D)
        assert weights.shape == (B, S, 1)

    def test_multiple_sources(self):
        module = LayerAttnRes(D, layer_idx=3)
        sources = [torch.randn(B, S, D) for _ in range(3)]
        out, weights = module(sources, return_weights=True)
        assert out.shape == (B, S, D)
        assert weights.shape == (B, S, 3)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B, S), atol=1e-5)

    def test_gradient_flow(self):
        module = LayerAttnRes(D, layer_idx=2)
        sources = [torch.randn(B, S, D, requires_grad=True) for _ in range(2)]
        out, _ = module(sources)
        loss = out.sum()
        loss.backward()
        for s in sources:
            assert s.grad is not None


class TestBlockAttnRes:
    def test_forward(self):
        module = BlockAttnRes(D, num_blocks=4, block_idx=2)
        blocks = [torch.randn(B, S, D) for _ in range(3)]
        out, weights = module(blocks, return_weights=True)
        assert out.shape == (B, S, D)
        assert weights.shape == (B, S, 3)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B, S), atol=1e-5)


class TestOnlineSoftmaxMerge:
    def test_matches_batch_softmax(self):
        merger = OnlineSoftmaxMerge(D)
        num_layers = 4
        outputs = [torch.randn(B, S, D) for _ in range(num_layers)]
        scores = [torch.randn(B, S) for _ in range(num_layers)]

        running_sum = outputs[0] * 1.0
        running_max = scores[0]
        running_exp_sum = torch.ones(B, S)

        for i in range(1, num_layers):
            running_sum, running_max, running_exp_sum = merger(
                running_sum, running_max, running_exp_sum,
                outputs[i], scores[i],
            )

        online_result = merger.finalize(running_sum, running_exp_sum)

        all_scores = torch.stack(scores, dim=-1)
        weights = torch.softmax(all_scores, dim=-1)
        stacked = torch.stack(outputs, dim=2)
        batch_result = (weights.unsqueeze(-1) * stacked).sum(dim=2)

        assert torch.allclose(online_result, batch_result, atol=1e-5)


# ---- Adaptive Transformer ----

class TestAdaptiveTransformer:
    def test_forward_basic(self):
        config = AdaptiveTransformerConfig(
            d_model=D, n_heads=4, n_layers=4, n_blocks=2,
            d_ff=D*4, vocab_size=100, max_seq_len=32,
        )
        model = AdaptiveTransformerWithAttnRes(config)
        input_ids = torch.randint(0, 100, (B, S))
        result = model(input_ids, return_routing_info=True)
        assert result["logits"].shape == (B, S, 100)
        assert len(result["routing_weights"]) > 0

    def test_skipping(self):
        config = AdaptiveTransformerConfig(
            d_model=D, n_heads=4, n_layers=8, n_blocks=4,
            d_ff=D*4, vocab_size=100, max_seq_len=32,
            enable_skipping=True, skip_threshold=0.5,
        )
        model = AdaptiveTransformerWithAttnRes(config)
        model.eval()
        input_ids = torch.randint(0, 100, (B, S))
        result = model(input_ids, return_routing_info=True)
        assert result["logits"].shape == (B, S, 100)

    def test_causal_lm_loss(self):
        config = AdaptiveTransformerConfig(
            d_model=D, n_heads=4, n_layers=4, n_blocks=2,
            d_ff=D*4, vocab_size=100, max_seq_len=32,
        )
        model = AdaptiveTransformerForCausalLM(config)
        input_ids = torch.randint(0, 100, (B, S))
        result = model(input_ids, labels=input_ids, return_routing_info=True)
        assert "loss" in result
        assert result["loss"].requires_grad

    def test_routing_statistics(self):
        config = AdaptiveTransformerConfig(
            d_model=D, n_heads=4, n_layers=4, n_blocks=2,
            d_ff=D*4, vocab_size=100, max_seq_len=32,
        )
        model = AdaptiveTransformerWithAttnRes(config)
        model.eval()
        input_ids = torch.randint(0, 100, (B, S))
        stats = model.get_routing_statistics(input_ids)
        assert "importance_matrix" in stats
        assert "block_importance" in stats
        assert len(stats["block_importance"]) > 0


# ---- Looping Transformer ----

class TestLoopingTransformer:
    def test_forward(self):
        config = LoopingTransformerConfig(
            d_model=D, n_heads=4, n_layers=4,
            n_unique_blocks=2, max_loop_depth=6,
            d_ff=D*4, vocab_size=100, max_seq_len=32,
            use_adaptive_halting=True,
        )
        model = LoopingTransformerWithAttnRes(config)
        input_ids = torch.randint(0, 100, (B, S))
        result = model(input_ids, return_routing_info=True)
        assert result["logits"].shape == (B, S, 100)
        assert "effective_depth" in result

    def test_with_loss(self):
        config = LoopingTransformerConfig(
            d_model=D, n_heads=4, n_layers=4,
            n_unique_blocks=2, max_loop_depth=6,
            d_ff=D*4, vocab_size=100, max_seq_len=32,
            use_adaptive_halting=True,
        )
        model = LoopingTransformerWithAttnRes(config)
        input_ids = torch.randint(0, 100, (B, S))
        result = model.forward_with_loss(input_ids, input_ids)
        assert "loss" in result
        assert "ponder_cost" in result
        assert result["loss"].requires_grad


# ---- VLA ----

class TestVLAAdaptive:
    def test_forward(self):
        config = VLAAdaptiveConfig(
            d_model=D, n_heads=4, n_layers=4, n_blocks=2,
            d_ff=D*4, vocab_size=100, max_seq_len=128,
            vision_dim=128, vision_seq_len=16,
            action_dim=7, action_chunk_size=8,
        )
        model = VLAAdaptiveTransformer(config)
        input_ids = torch.randint(0, 100, (B, 16))
        vision = torch.randn(B, 16, 128)
        target_actions = torch.randn(B, 8, 7)

        result = model(input_ids, vision, target_actions=target_actions, return_routing_info=True)
        assert result["actions"].shape == (B, 8, 7)
        assert "action_loss" in result

    def test_modality_analysis(self):
        config = VLAAdaptiveConfig(
            d_model=D, n_heads=4, n_layers=4, n_blocks=2,
            d_ff=D*4, vocab_size=100, max_seq_len=128,
            vision_dim=128, vision_seq_len=16,
            action_dim=7, action_chunk_size=8,
        )
        model = VLAAdaptiveTransformer(config)
        input_ids = torch.randint(0, 100, (B, 16))
        vision = torch.randn(B, 16, 128)

        analysis = model.analyze_modality_depth(input_ids, vision)
        assert "VISION" in analysis
        assert "LANGUAGE" in analysis


# ---- Data Pipeline ----

class TestData:
    def test_structured_synthetic(self):
        ds = StructuredSyntheticLM(vocab_size=100, seq_len=32, num_samples=100)
        assert len(ds) == 100
        sample = ds[0]
        assert sample["input_ids"].shape == (32,)
        assert sample["labels"].shape == (32,)
        assert "difficulty" in sample

    def test_structured_vla(self):
        ds = StructuredVLADataset(
            vision_dim=64, vision_seq_len=8, lang_seq_len=8,
            action_dim=7, action_chunk_size=4, vocab_size=100,
            num_samples=50,
        )
        assert len(ds) == 50
        sample = ds[0]
        assert sample["vision_features"].shape == (8, 64)
        assert sample["input_ids"].shape == (8,)
        assert sample["target_actions"].shape == (4, 7)
        assert "task_type" in sample

    def test_create_dataloaders(self):
        train_loader, val_loader, vocab = create_lm_dataloaders(
            dataset_type="structured_synthetic",
            batch_size=8, seq_len=32, vocab_size=100,
            num_train=100, num_val=50,
        )
        assert vocab == 100
        batch = next(iter(train_loader))
        assert batch["input_ids"].shape[0] == 8
        assert batch["input_ids"].shape[1] == 32

    def test_difficulty_distribution(self):
        ds = StructuredSyntheticLM(vocab_size=100, seq_len=32, num_samples=90, difficulty_mix="mixed")
        diffs = ds.difficulties
        assert (diffs == 0).sum() == 30  # Easy
        assert (diffs == 1).sum() == 30  # Medium
        assert (diffs == 2).sum() == 30  # Hard


# ---- Utilities ----

class TestUtils:
    def test_flop_counting(self):
        flops = count_transformer_flops(
            d_model=256, n_heads=8, d_ff=1024,
            seq_len=128, n_layers=4, vocab_size=1000,
        )
        assert flops["total"] > 0
        assert flops["total_gflops"] > 0
        assert flops["per_layer_total"] > 0

    def test_effective_flops(self):
        config = AdaptiveTransformerConfig(
            d_model=256, n_heads=8, n_layers=8, n_blocks=4,
            d_ff=1024, vocab_size=1000, max_seq_len=128,
        )
        result = count_effective_flops(config, seq_len=128, blocks_executed=2)
        assert 0 < result["flops_ratio"] < 1.0
        assert result["blocks_executed"] == 2
        assert result["total_blocks"] == 4

    def test_cosine_warmup_scheduler(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=100)

        # During warmup, LR should increase
        scheduler.step(0)
        lr0 = optimizer.param_groups[0]["lr"]
        scheduler.step(5)
        lr5 = optimizer.param_groups[0]["lr"]
        assert lr5 > lr0

        # After warmup, LR should decrease
        scheduler.step(10)
        lr10 = optimizer.param_groups[0]["lr"]
        scheduler.step(50)
        lr50 = optimizer.param_groups[0]["lr"]
        assert lr50 < lr10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
