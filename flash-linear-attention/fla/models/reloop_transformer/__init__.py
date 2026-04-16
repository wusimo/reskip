from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.reloop_transformer.configuration_reloop_transformer import ReLoopTransformerConfig
from fla.models.reloop_transformer.modeling_reloop_transformer import (
    ReLoopTransformerForCausalLM,
    ReLoopTransformerModel,
)

AutoConfig.register(ReLoopTransformerConfig.model_type, ReLoopTransformerConfig, exist_ok=True)
AutoModel.register(ReLoopTransformerConfig, ReLoopTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(ReLoopTransformerConfig, ReLoopTransformerForCausalLM, exist_ok=True)

__all__ = [
    "ReLoopTransformerConfig",
    "ReLoopTransformerForCausalLM",
    "ReLoopTransformerModel",
]
