# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 153716 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn
from src.starvla_integration import StarVLABackboneSkipContext


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model — resolve action token range from the
        # tokenizer at load time (2B and 4B have different vocab sizes so the
        # added robot_action_* tokens land at different ID ranges).
        if "-Action" in model_id:
            tok = processor.tokenizer
            start_id = tok.convert_tokens_to_ids("<robot_action_0>")
            if start_id is None or start_id == tok.unk_token_id:
                self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
                self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX
            else:
                last_id = start_id
                for candidate in range(start_id + 2047, start_id, -1):
                    tok_str = tok.convert_ids_to_tokens(candidate)
                    if tok_str and tok_str.startswith("<robot_action_"):
                        last_id = candidate
                        break
                self._ACTION_TOKEN_MIN = int(start_id)
                self._ACTION_TOKEN_MAX = int(last_id)

    def get_hidden_size(self) -> int:
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        return int(getattr(self.model.config, "hidden_size"))

    def get_num_hidden_layers(self) -> int:
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            return int(text_cfg.num_hidden_layers)
        return int(getattr(self.model.config, "num_hidden_layers"))

    def get_vision_token_ids(self) -> tuple[int, int]:
        return (IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX)

    def get_action_token_range(self) -> Optional[tuple[int, int]]:
        if hasattr(self, "_ACTION_TOKEN_MIN") and hasattr(self, "_ACTION_TOKEN_MAX"):
            return (self._ACTION_TOKEN_MIN, self._ACTION_TOKEN_MAX)
        return None

    def get_text_layers(self):
        return self.model.language_model.layers

    def forward_with_attnres_skip(
        self,
        adapter,
        *,
        enable_skipping: bool = False,
        dynamic_skip_config: Optional[dict] = None,
        **kwargs,
    ):
        """Run Qwen3-VL with per-block AttnRes in-backbone.

        Args:
            adapter: StarVLAAttnResAdapter holding router/adapters/γ.
            enable_skipping: when True, per-block dyn-skip is consulted;
                otherwise every block runs its transformer layers.
            dynamic_skip_config: retrofit-style schema — see
                ``StarVLABackboneSkipContext`` docstring. ``use_cache=True``
                is supported; skipped blocks populate their KV cache via
                a K/V-only path.
            **kwargs: forwarded to ``self.model(...)`` (input_ids,
                attention_mask, pixel_values, past_key_values, use_cache, ...).
        """
        controller = StarVLABackboneSkipContext(
            adapter,
            enable_skipping=enable_skipping,
            dynamic_skip_config=dynamic_skip_config,
        )
        # Bind the language_model so the context can monkey-patch its forward.
        controller.bind_text_model(self.model.model.language_model)
        with controller, torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )
        return outputs, controller.get_summary()

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now.
            # Prefer tokenizer-resolved range (works for both 2B [151669, 153716]
            # and 4B [151936, 153983]); fall back to hardcoded 2B range.
            action_token_min = getattr(self, "_ACTION_TOKEN_MIN", _ACTION_TOKEN_MIN)
            action_token_max = getattr(self, "_ACTION_TOKEN_MAX", _ACTION_TOKEN_MAX)
            labels = batch_inputs['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning (f"action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md.")
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)




if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
