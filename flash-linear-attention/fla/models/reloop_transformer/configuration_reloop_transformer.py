import warnings

from transformers.configuration_utils import PretrainedConfig


class ReLoopTransformerConfig(PretrainedConfig):
    model_type = "reloop_transformer"
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
        num_recurrent_blocks: int | None = None,
        max_loops: int = 1,
        halt_mode: str = "head",
        halt_threshold: float = 0.99,
        attnres_halt_threshold: float = 0.15,
        attnres_halt_temperature: float = 0.05,
        halt_kl_weight: float = 0.0,
        halt_kl_min_weight: float = 0.0,
        halt_kl_decay_steps: int = 0,
        halt_use_phase1_stats: bool = True,
        halt_detach_phase1_stats: bool = True,
        halt_use_position_bias: bool = True,
        training_soft_min_halt: bool = True,
        training_full_depth: bool = False,
        halt_curriculum_disable_after_target: bool = True,
        halt_retain_target_after_curriculum: bool = False,
        early_exit_penalty_weight: float = 0.0,
        early_exit_penalty_warmup_steps: int = 0,
        focused_halt_loss_weight: float = 0.0,
        focused_halt_loss_start_step: int = 0,
        focused_halt_loss_warmup_steps: int = 0,
        focused_halt_improvement_margin: float = 0.0,
        focused_halt_target_temperature: float = 0.1,
        focused_halt_num_tokens: int = 128,
        multi_exit_loss_weight: float = 0.0,
        ponder_loss_weight: float = 0.0,
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
        # ReLoop is always the looping variant.
        self.enable_looping = True
        self.num_recurrent_blocks = num_recurrent_blocks
        self.max_loops = max_loops
        self.halt_mode = halt_mode
        self.halt_threshold = halt_threshold
        self.attnres_halt_threshold = attnres_halt_threshold
        self.attnres_halt_temperature = attnres_halt_temperature
        self.halt_kl_weight = halt_kl_weight
        self.halt_kl_min_weight = halt_kl_min_weight
        self.halt_kl_decay_steps = halt_kl_decay_steps
        self.halt_use_phase1_stats = halt_use_phase1_stats
        self.halt_detach_phase1_stats = halt_detach_phase1_stats
        self.halt_use_position_bias = halt_use_position_bias
        self.training_soft_min_halt = training_soft_min_halt
        self.training_full_depth = training_full_depth
        self.halt_curriculum_disable_after_target = halt_curriculum_disable_after_target
        self.halt_retain_target_after_curriculum = halt_retain_target_after_curriculum
        self.early_exit_penalty_weight = early_exit_penalty_weight
        self.early_exit_penalty_warmup_steps = early_exit_penalty_warmup_steps
        self.focused_halt_loss_weight = focused_halt_loss_weight
        self.focused_halt_loss_start_step = focused_halt_loss_start_step
        self.focused_halt_loss_warmup_steps = focused_halt_loss_warmup_steps
        self.focused_halt_improvement_margin = focused_halt_improvement_margin
        self.focused_halt_target_temperature = focused_halt_target_temperature
        self.focused_halt_num_tokens = focused_halt_num_tokens
        self.multi_exit_loss_weight = multi_exit_loss_weight
        self.ponder_loss_weight = ponder_loss_weight
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
        # Allow `num_recurrent_blocks=None` at default construction so that
        # HuggingFace utilities like `to_diff_dict()` can instantiate the
        # class with no args. In that case fall back to `attn_res_num_blocks`
        # (i.e. no actual weight sharing). Real ReLoop configs must set both
        # `num_recurrent_blocks` and `max_loops`.
        if num_recurrent_blocks is None:
            num_recurrent_blocks = attn_res_num_blocks
            self.num_recurrent_blocks = attn_res_num_blocks
        if num_recurrent_blocks <= 0 or max_loops <= 0:
            raise ValueError("`num_recurrent_blocks` and `max_loops` must be positive.")
        if not 0.0 < halt_threshold <= 1.0:
            raise ValueError("`halt_threshold` must be in (0, 1].")
        if halt_kl_weight < 0 or halt_kl_min_weight < 0:
            raise ValueError("`halt_kl_weight` and `halt_kl_min_weight` must be non-negative.")
        if halt_kl_decay_steps < 0:
            raise ValueError("`halt_kl_decay_steps` must be non-negative.")
        if not isinstance(halt_use_phase1_stats, bool):
            raise ValueError("`halt_use_phase1_stats` must be a boolean.")
        if not isinstance(halt_detach_phase1_stats, bool):
            raise ValueError("`halt_detach_phase1_stats` must be a boolean.")
        if not isinstance(training_soft_min_halt, bool):
            raise ValueError("`training_soft_min_halt` must be a boolean.")
        if early_exit_penalty_weight < 0:
            raise ValueError("`early_exit_penalty_weight` must be non-negative.")
        if early_exit_penalty_warmup_steps < 0:
            raise ValueError("`early_exit_penalty_warmup_steps` must be non-negative.")
        if focused_halt_loss_weight < 0:
            raise ValueError("`focused_halt_loss_weight` must be non-negative.")
        if focused_halt_loss_start_step < 0:
            raise ValueError("`focused_halt_loss_start_step` must be non-negative.")
        if focused_halt_loss_warmup_steps < 0:
            raise ValueError("`focused_halt_loss_warmup_steps` must be non-negative.")
        if focused_halt_target_temperature <= 0:
            raise ValueError("`focused_halt_target_temperature` must be positive.")
        if focused_halt_num_tokens < 0:
            raise ValueError("`focused_halt_num_tokens` must be non-negative.")
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
                "`attn_res_num_blocks` must equal `num_recurrent_blocks * max_loops` for ReLoop."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
