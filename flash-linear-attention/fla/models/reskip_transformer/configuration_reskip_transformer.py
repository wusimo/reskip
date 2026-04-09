import warnings

from transformers.configuration_utils import PretrainedConfig


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
        enable_looping: bool = False,
        num_recurrent_blocks: int | None = None,
        max_loops: int = 1,
        enable_skip_inference: bool = False,
        skip_keep_mask: list[int] | list[bool] | None = None,
        dynamic_skip_strategy: str | None = None,
        dynamic_skip_threshold: float | None = None,
        dynamic_skip_position_thresholds: list[float] | None = None,
        dynamic_skip_max_skips: int | None = None,
        halt_threshold: float = 0.99,
        halt_kl_weight: float = 0.0,
        halt_kl_min_weight: float = 0.0,
        halt_kl_decay_steps: int = 0,
        ponder_loss_weight: float = 0.0,
        routing_regularization_weight: float = 0.0,
        routing_entropy_weight: float = 0.0,
        routing_entropy_target: float = 0.0,
        routing_entropy_warmup_steps: int = 0,
        ponder_loss_warmup_steps: int = 0,
        ponder_budget_start_step: int = 0,
        ponder_target_depth_ratio: float = 0.5,
        ponder_target_steps: int = 0,
        **kwargs,
    ):
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
        self.enable_looping = enable_looping
        self.num_recurrent_blocks = num_recurrent_blocks
        self.max_loops = max_loops
        self.enable_skip_inference = enable_skip_inference
        self.skip_keep_mask = list(skip_keep_mask) if skip_keep_mask is not None else None
        self.dynamic_skip_strategy = dynamic_skip_strategy
        self.dynamic_skip_threshold = dynamic_skip_threshold
        self.dynamic_skip_position_thresholds = (
            list(dynamic_skip_position_thresholds) if dynamic_skip_position_thresholds is not None else None
        )
        self.dynamic_skip_max_skips = dynamic_skip_max_skips
        self.halt_threshold = halt_threshold
        self.halt_kl_weight = halt_kl_weight
        self.halt_kl_min_weight = halt_kl_min_weight
        self.halt_kl_decay_steps = halt_kl_decay_steps
        self.ponder_loss_weight = ponder_loss_weight
        self.routing_regularization_weight = routing_regularization_weight
        self.routing_entropy_weight = routing_entropy_weight
        self.routing_entropy_target = routing_entropy_target
        self.routing_entropy_warmup_steps = routing_entropy_warmup_steps
        self.ponder_loss_warmup_steps = ponder_loss_warmup_steps
        self.ponder_budget_start_step = ponder_budget_start_step
        self.ponder_target_depth_ratio = ponder_target_depth_ratio
        self.ponder_target_steps = ponder_target_steps

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time."
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled. If training becomes unstable, disable it first."
            )
        if num_hidden_layers % attn_res_num_blocks != 0:
            raise ValueError(
                f"`num_hidden_layers` ({num_hidden_layers}) must be divisible by "
                f"`attn_res_num_blocks` ({attn_res_num_blocks})."
            )
        if routing_entropy_weight < 0:
            raise ValueError("`routing_entropy_weight` must be non-negative.")
        if routing_entropy_target < 0:
            raise ValueError("`routing_entropy_target` must be non-negative.")
        if routing_entropy_warmup_steps < 0:
            raise ValueError("`routing_entropy_warmup_steps` must be non-negative.")
        if routing_regularization_weight < 0:
            raise ValueError("`routing_regularization_weight` must be non-negative.")
        if (
            routing_regularization_weight > 0
            and (routing_entropy_weight > 0 or routing_entropy_target > 0 or routing_entropy_warmup_steps > 0)
        ):
            warnings.warn(
                "`routing_regularization_weight` is enabled, so legacy routing entropy target/warmup settings "
                "will be ignored during training."
            )
        if enable_looping:
            if num_recurrent_blocks is None:
                raise ValueError("`num_recurrent_blocks` is required when `enable_looping=True`.")
            if num_recurrent_blocks <= 0 or max_loops <= 0:
                raise ValueError("`num_recurrent_blocks` and `max_loops` must be positive.")
            if not 0.0 < halt_threshold <= 1.0:
                raise ValueError("`halt_threshold` must be in (0, 1].")
            if halt_kl_weight < 0 or halt_kl_min_weight < 0:
                raise ValueError("`halt_kl_weight` and `halt_kl_min_weight` must be non-negative.")
            if halt_kl_decay_steps < 0:
                raise ValueError("`halt_kl_decay_steps` must be non-negative.")
            if ponder_loss_weight < 0:
                raise ValueError("`ponder_loss_weight` must be non-negative.")
            if ponder_loss_warmup_steps < 0:
                raise ValueError("`ponder_loss_warmup_steps` must be non-negative.")
            if ponder_budget_start_step < 0:
                raise ValueError("`ponder_budget_start_step` must be non-negative.")
            if not 0.0 < ponder_target_depth_ratio <= 1.0:
                raise ValueError("`ponder_target_depth_ratio` must be in (0, 1].")
            if ponder_target_steps < 0:
                raise ValueError("`ponder_target_steps` must be non-negative.")
            if attn_res_num_blocks != num_recurrent_blocks * max_loops:
                raise ValueError(
                    "`attn_res_num_blocks` must equal `num_recurrent_blocks * max_loops` in looping mode."
                )
        elif num_recurrent_blocks is None:
            self.num_recurrent_blocks = attn_res_num_blocks

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
