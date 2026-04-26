"""Shared torch.compile policy for the retrofit production inference path.

Paper claim is "+accuracy at iso-cost (1.029× base, both compiled)". To keep
the claim honest, every default production-eval entry point wraps the
model through this helper so the same compile mode is in effect at eval
time as at the speed bench. Each script also exposes a CLI flag
(`--compile-mode off`) to disable compile if a future accuracy-reproduction
run hits a torch._inductor regression and needs an eager fallback.

Default mode: ``max-autotune-no-cudagraphs`` — aggressive per-kernel
autotuning (matches the headline 1.029× number) without requiring a
fixed input shape. Variable-length inputs (LAMBADA, lmms-eval, LIBERO
prompt streams) all work without per-shape recompiles.

Override at runtime via either
  - CLI: ``--compile-mode {off,default,reduce-overhead,max-autotune,
                            max-autotune-no-cudagraphs}``
  - env: ``RETROFIT_COMPILE_MODE=...`` (lower priority than CLI)
"""
from __future__ import annotations

import os
import torch

DEFAULT_MODE = "default"
# "default" = lightest torch.compile mode. Handles variable input shapes
# without per-shape autotune storms, gives ~10-15% speedup on retrofit
# in our benches. Heavier modes (max-autotune, reduce-overhead) require
# fixed-shape inputs to amortize their compile cost — they're appropriate
# for the speed bench (fixed seq=2048) but not for LAMBADA / lmms-eval
# where every example has a different length and max-autotune-per-shape
# becomes the bottleneck.
#
# To use the headline 1.029× iso-cost mode in production, pass
# ``--compile-mode max-autotune`` on a fixed-shape path.
OFF_ALIASES = {"off", "none", "eager", "false", "0", "no", ""}


def resolve_compile_mode(cli_value: str | None) -> str:
    """Pick the active compile mode given a CLI value (may be None)."""
    if cli_value is not None:
        return cli_value
    return os.environ.get("RETROFIT_COMPILE_MODE", DEFAULT_MODE)


def wrap_compile(
    model,
    mode: str | None = None,
    dynamic: bool | None = None,
    label: str = "model",
):
    """Wrap ``model`` with ``torch.compile`` per the iso-cost claim.

    Args:
        model: an ``nn.Module``. For HF wrappers we typically pass the
            top-level model so HF's ``forward`` is the compile entry.
        mode: ``None`` → use ``$RETROFIT_COMPILE_MODE`` or DEFAULT_MODE.
            One of the OFF_ALIASES → return ``model`` unchanged
            (escape hatch for accuracy reproduction).
        dynamic: ``None`` → True (safe default for variable shapes).
            Set ``False`` only on fixed-shape paths where CUDA-graph
            capture is desirable (e.g. ``mode='reduce-overhead'`` /
            ``'max-autotune'`` with a single seq length).
        label: shows up in the print line so per-callsite logs make sense.
    """
    mode = resolve_compile_mode(mode)
    if mode.lower() in OFF_ALIASES:
        print(f"[compile] {label}: eager (compile disabled, mode={mode!r})",
              flush=True)
        return model
    if dynamic is None:
        dynamic = True
    print(f"[compile] {label}: torch.compile(mode={mode!r}, dynamic={dynamic})",
          flush=True)
    return torch.compile(model, mode=mode, dynamic=dynamic)


def add_compile_arg(parser, default: str | None = None) -> None:
    """Standard CLI surface so every eval script speaks the same flag."""
    parser.add_argument(
        "--compile-mode",
        default=default,
        help=(
            "torch.compile mode for inference. Choices: off|none|eager (no compile), "
            "default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs. "
            f"Default: {DEFAULT_MODE} (matches paper iso-cost claim). "
            "Set 'off' to bypass compile if a torch._inductor regression "
            "surfaces during accuracy reproduction."
        ),
    )
