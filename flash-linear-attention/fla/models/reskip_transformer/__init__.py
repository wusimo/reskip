from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.reskip_transformer.configuration_reskip_transformer import ReSkipTransformerConfig
from fla.models.reskip_transformer.modeling_reskip_transformer import (
    ReSkipTransformerForCausalLM,
    ReSkipTransformerModel,
)

AutoConfig.register(ReSkipTransformerConfig.model_type, ReSkipTransformerConfig, exist_ok=True)
AutoModel.register(ReSkipTransformerConfig, ReSkipTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(ReSkipTransformerConfig, ReSkipTransformerForCausalLM, exist_ok=True)

__all__ = [
    "ReSkipTransformerConfig",
    "ReSkipTransformerForCausalLM",
    "ReSkipTransformerModel",
]
