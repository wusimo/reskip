"""lmms-eval model plugin for the AR-Retrofit on top of Qwen3-VL-2B.

Extends the stock ``Qwen3_VL`` plugin: after the base model is loaded the
retrofit wrapper monkey-patches ``base.model.language_model.forward`` in
place, so ``self.model.generate(...)`` (called by lmms-eval) executes through
our retrofit. Usage::

    python -m retrofit.lmms_eval_retrofit \
        --retrofit_state outputs/H_r256_5k/retrofit_attnres_state.pt \
        --model_args "pretrained=/home/user01/Minko/models/Qwen3-VL-2B,retrofit_state_path=..." \
        --tasks mmbench_en_dev,mmstar,mmmu_val,ai2d,ocrbench,mathvista_testmini,realworldqa,hallusion_bench \
        ...

Dynamic-skip support: pass ``dynamic_skip=1,quantile=0.95,max_skips=1,positions=4,6,11``
in ``--model_args`` to turn on phase-1 skip during generation. Thresholds are
calibrated on 32 LAMBADA prefixes once at load time.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # retrofit/eval (this plugin)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # retrofit/ (core module)

import torch
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
from lmms_eval.api.registry import register_model
from lmms_eval.models.registry_v2 import ModelManifest
from lmms_eval.models import MODEL_REGISTRY_V2

from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


@register_model("qwen3_vl_retrofit")
class Qwen3_VL_Retrofit(Qwen3_VL):
    """Qwen3-VL base with AR-Retrofit state loaded; monkey-patches forward."""

    def __init__(
        self,
        retrofit_state_path: str = None,
        num_blocks: int = 14,
        adapter_rank: int = 256,
        dynamic_skip: bool = False,
        dyn_quantile: float = 0.95,
        dyn_max_skips: int = 1,
        dyn_positions: str = "4,6,11",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if retrofit_state_path is None:
            raise ValueError("retrofit_state_path is required for qwen3_vl_retrofit")
        self._retrofit = self._load_retrofit(
            retrofit_state_path, num_blocks, adapter_rank
        )
        if dynamic_skip:
            self._configure_dynamic_skip(
                quantile=float(dyn_quantile),
                max_skips=int(dyn_max_skips),
                positions=tuple(int(x) for x in str(dyn_positions).split(",")),
            )

    def _load_retrofit(self, state_path, num_blocks, adapter_rank):
        ck = torch.load(state_path, map_location="cpu")
        cfg = ck.get("config", {})
        kwargs = {"num_blocks": int(cfg.get("num_blocks", num_blocks))}
        ar = cfg.get("adapter_rank", adapter_rank)
        kwargs["adapter_rank"] = int(ar)
        no_adapter = cfg.get("no_adapter", not ck.get("adapters"))
        if no_adapter:
            kwargs["no_adapter"] = True
            kwargs.pop("adapter_rank", None)

        model = self._model
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        wrapper = Qwen3VLAttnResRetrofit(model, **kwargs).to(device=device, dtype=dtype)
        wrapper.router.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()}
        )
        if not no_adapter:
            wrapper.adapters.load_state_dict(
                {k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()}
            )
        wrapper.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
        wrapper.eval()
        gmax = float(wrapper.gamma.detach().abs().max())
        gmean = float(wrapper.gamma.detach().mean())
        print(
            f"[retrofit] state loaded: num_blocks={kwargs['num_blocks']}, "
            f"adapter_rank={kwargs.get('adapter_rank', 'Identity')}, "
            f"γ_max={gmax:.3f}, γ_mean={gmean:+.3f}",
            flush=True,
        )
        return wrapper

    def _configure_dynamic_skip(self, quantile, max_skips, positions):
        """Calibrate per-block thresholds on 32 LAMBADA prefixes, then enable."""
        from datasets import load_dataset
        from collections import defaultdict

        retro = self._retrofit
        tok = self._tokenizer
        device = retro.gamma.device
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test").select(
            range(32)
        )
        samples = defaultdict(list)
        with torch.no_grad():
            for ex in ds:
                ids = tok.encode(
                    ex["text"].strip(), add_special_tokens=False
                )[:512]
                inp = torch.tensor([ids], device=device)
                out = retro(input_ids=inp, return_alpha=True)
                for bidx, trace in enumerate(out.skip_trace or []):
                    w = trace.get("w_recent")
                    if w is not None:
                        samples[bidx].append(w)
        thresholds = {}
        for b, vals in samples.items():
            if not vals:
                continue
            vs = sorted(vals)
            thresholds[b] = vs[int(quantile * (len(vs) - 1))]
        retro._dynamic_skip_config = dict(
            strategy="recent_weight_gt",
            thresholds=thresholds,
            positions=positions,
            max_skips=max_skips,
        )
        print(
            f"[retrofit] dynamic skip armed: q={quantile}, M={max_skips}, "
            f"P={positions}, calibrated thresholds={thresholds}",
            flush=True,
        )


# Register the model in MODEL_REGISTRY_V2 so lmms-eval resolves it by name.
MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="qwen3_vl_retrofit",
        simple_class_path="lmms_eval_retrofit.Qwen3_VL_Retrofit",
    ),
    overwrite=True,
)


if __name__ == "__main__":
    # Re-enter lmms-eval's CLI with our plugin registered.
    import lmms_eval.__main__ as m
    m.cli_evaluate()
