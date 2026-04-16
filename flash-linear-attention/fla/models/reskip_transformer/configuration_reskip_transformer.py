from transformers.configuration_utils import PretrainedConfig


_LEGACY_RELOOP_KEYS = {
    "enable_looping",
    "num_recurrent_blocks",
    "max_loops",
    "halt_threshold",
    "halt_kl_weight",
    "halt_kl_min_weight",
    "halt_kl_decay_steps",
    "halt_use_phase1_stats",
    "halt_detach_phase1_stats",
    "halt_use_position_bias",
    "training_soft_min_halt",
    "halt_curriculum_disable_after_target",
    "early_exit_penalty_weight",
    "early_exit_penalty_warmup_steps",
    "focused_halt_loss_weight",
    "focused_halt_loss_start_step",
    "focused_halt_loss_warmup_steps",
    "focused_halt_improvement_margin",
    "focused_halt_target_temperature",
    "focused_halt_num_tokens",
    "ponder_loss_weight",
    "ponder_loss_warmup_steps",
    "ponder_budget_start_step",
    "ponder_target_depth_ratio",
    "ponder_target_steps",
}

_LEGACY_SKIP_KEYS = {
    "dynamic_skip_granularity",
    "dynamic_skip_mlp_position_thresholds",
    "dynamic_skip_max_mlp_skips",
    "dynamic_skip_sample_quantile",
}


class ReSkipTransformerConfig(PretrainedConfig):
    model_type = "reskip_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        attn_res_num_blocks: int = 8,
        attn_res_temperature: float = 1.0,
        enable_skip_inference: bool = False,
        skip_keep_mask: list[int] | list[bool] | None = None,
        dynamic_skip_strategy: str | None = None,
        dynamic_skip_probe_mode: str | None = "all",
        dynamic_skip_threshold: float | None = None,
        dynamic_skip_position_thresholds: list[float] | None = None,
        dynamic_skip_max_skips: int | None = None,
        **kwargs,
    ):
        legacy_enable_looping = bool(kwargs.pop("enable_looping", False))
        kwargs.pop("num_recurrent_blocks", None)
        for key in list(kwargs):
            if key in _LEGACY_RELOOP_KEYS or key in _LEGACY_SKIP_KEYS:
                kwargs.pop(key)

        if legacy_enable_looping:
            raise ValueError(
                "`enable_looping=True` is no longer supported by `reskip_transformer`. "
                "Use `reloop_transformer` instead."
            )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.attn_res_num_blocks = attn_res_num_blocks
        self.attn_res_temperature = attn_res_temperature

        self.enable_looping = False
        self.num_recurrent_blocks = attn_res_num_blocks

        self.enable_skip_inference = enable_skip_inference
        self.skip_keep_mask = list(skip_keep_mask) if skip_keep_mask is not None else None
        self.dynamic_skip_strategy = dynamic_skip_strategy
        self.dynamic_skip_probe_mode = dynamic_skip_probe_mode
        self.dynamic_skip_threshold = dynamic_skip_threshold
        self.dynamic_skip_position_thresholds = (
            list(dynamic_skip_position_thresholds) if dynamic_skip_position_thresholds is not None else None
        )
        self.dynamic_skip_max_skips = dynamic_skip_max_skips

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time."
            )
        if fuse_linear_cross_entropy:
            import warnings

            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled. If training becomes unstable, disable it first."
            )
        if num_hidden_layers % attn_res_num_blocks != 0:
            raise ValueError(
                f"`num_hidden_layers` ({num_hidden_layers}) must be divisible by "
                f"`attn_res_num_blocks` ({attn_res_num_blocks})."
            )
        if self.skip_keep_mask is not None and len(self.skip_keep_mask) != attn_res_num_blocks:
            raise ValueError(
                f"`skip_keep_mask` length ({len(self.skip_keep_mask)}) must match "
                f"`attn_res_num_blocks` ({attn_res_num_blocks})."
            )
        if (
            self.dynamic_skip_position_thresholds is not None
            and len(self.dynamic_skip_position_thresholds) != attn_res_num_blocks
        ):
            raise ValueError(
                f"`dynamic_skip_position_thresholds` length ({len(self.dynamic_skip_position_thresholds)}) must match "
                f"`attn_res_num_blocks` ({attn_res_num_blocks})."
            )
        if self.dynamic_skip_max_skips is not None and self.dynamic_skip_max_skips < 0:
            raise ValueError("`dynamic_skip_max_skips` must be non-negative.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
