"""Multi-source data loaders for retrofit v2.

Unified sample format yielded by every dataset iterator:
    {"messages": [...Qwen3VL-style dicts...], "images": [PIL.Image, ...]}

The `messages` content is either a string (pure text sample) or a list of
content dicts with `{type: image|text}` (multimodal sample), matching the
Qwen3VL chat template expectations.

Adapters handle local files downloaded via
    hf download <repo> --repo-type dataset --local-dir /home/user01/Minko/datasets/<name>
The data root and the mix spec are configurable from the training CLI.

All streams are restart-safe: when exhausted, the mixer re-initialises the
iterator with a new seed.
"""
from __future__ import annotations

import glob
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from datasets import load_dataset
from PIL import Image

DATA_ROOT = Path("/home/user01/Minko/datasets")
SHAREGPT_ROLE = {"human": "user", "gpt": "assistant", "user": "user",
                 "assistant": "assistant", "system": "system"}


# ----------------------- helpers ----------------------- #

def _text_sample(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "images": [],
    }


def _vl_sample(user_text: str, assistant: str, images: list) -> dict:
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": user_text})
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
        ],
        "images": list(images),
    }


def _sharegpt_to_messages(convs, has_image: bool = False,
                           system: str | None = None) -> list[dict] | None:
    """Convert ShareGPT [{from, value}, ...] to Qwen3VL messages.
    Strips any literal `<image>` placeholder from the first user turn —
    Qwen3VL template injects vision markers via content-list instead."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    first_user_seen = False
    for m in convs:
        role = SHAREGPT_ROLE.get((m.get("from") or m.get("role") or "").lower())
        text = m.get("value") or m.get("content")
        if role is None or text is None:
            continue
        if role == "user" and not first_user_seen:
            first_user_seen = True
            text = text.replace("<image>\n", "").replace("<image>", "").strip()
            if has_image:
                content = [{"type": "image"}, {"type": "text", "text": text}]
                msgs.append({"role": "user", "content": content})
                continue
        msgs.append({"role": role, "content": text})
    # at least one user+assistant pair
    if sum(1 for m in msgs if m["role"] == "user") == 0:
        return None
    if sum(1 for m in msgs if m["role"] == "assistant") == 0:
        return None
    return msgs


def _pil_from_field(field):
    """Accept PIL.Image, dict({bytes, path}), or raw bytes."""
    if field is None:
        return None
    if isinstance(field, Image.Image):
        return field
    if isinstance(field, dict):
        b = field.get("bytes")
        if b:
            return Image.open(io.BytesIO(b)).convert("RGB")
        p = field.get("path")
        if p and Path(p).is_file():
            return Image.open(p).convert("RGB")
    if isinstance(field, (bytes, bytearray)):
        return Image.open(io.BytesIO(field)).convert("RGB")
    return None


def _iter_parquet(glob_pat: str, seed: int, buffer: int = 500):
    files = sorted(glob.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No parquet matches: {glob_pat}")
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer)
    return ds


def _iter_jsonl(path: str, seed: int, buffer: int = 500):
    ds = load_dataset("json", data_files=[path], split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer)
    return ds


def _iter_big_json(path: str, seed: int):
    """Standard JSON array (not JSONL). Load into memory, shuffle indices."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    for i in idx:
        yield rows[i]


# ----------------------- per-dataset adapters ----------------------- #

def iter_ultrachat(seed: int):
    pat = str(DATA_ROOT / "ultrachat_200k/HuggingFaceH4___ultrachat_200k"
              "/*/train_sft-*.arrow")
    files = sorted(glob.glob(pat))
    if files:
        ds = load_dataset("arrow", data_files=files, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=500)
    else:
        # fallback to hub-cached form
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          cache_dir=str(DATA_ROOT / "ultrachat_200k"),
                          split="train_sft", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=500)
    for s in ds:
        msgs = s.get("messages")
        if msgs:
            yield {"messages": msgs, "images": []}


def iter_llava_vsft(seed: int):
    # existing local cache is an HF cache dir; use the same load_dataset path
    ds = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft",
                      cache_dir=str(DATA_ROOT / "llava_instruct_vsft"),
                      split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx:
        s = ds[i]
        msgs = s.get("messages")
        imgs = s.get("images") or []
        if msgs:
            yield {"messages": msgs, "images": imgs}


def iter_numina_math(seed: int):
    ds = _iter_parquet(str(DATA_ROOT / "NuminaMath-CoT/data/train-*.parquet"), seed)
    for s in ds:
        msgs = s.get("messages")
        if msgs and len(msgs) >= 2:
            yield {"messages": msgs, "images": []}


def iter_open_thoughts(seed: int):
    ds = _iter_parquet(str(DATA_ROOT / "OpenThoughts-114k/data/train-*.parquet"), seed)
    for s in ds:
        convs = s.get("conversations")
        if not convs:
            continue
        sys_txt = s.get("system")
        msgs = _sharegpt_to_messages(convs, has_image=False, system=sys_txt)
        if msgs:
            yield {"messages": msgs, "images": []}


def iter_openmath2(seed: int):
    ds = _iter_parquet(str(DATA_ROOT / "OpenMathInstruct-2/data/*.parquet"), seed)
    for s in ds:
        p = s.get("problem"); sol = s.get("generated_solution")
        ans = s.get("expected_answer")
        if not (p and sol):
            continue
        # Append the expected answer marker so the model sees the label target.
        if ans and f"answer is {ans}" not in sol and f"= {ans}" not in sol[-40:]:
            sol = f"{sol}\n\nThe answer is {ans}."
        yield _text_sample(p, sol)


def iter_long_alpaca(seed: int):
    path = str(DATA_ROOT / "LongAlpaca-12k/LongAlpaca-12k.json")
    for row in _iter_big_json(path, seed):
        instr = row.get("instruction") or ""
        inp = row.get("input") or ""
        out = row.get("output") or ""
        if not (instr and out):
            continue
        user = instr if not inp else f"{instr}\n\n{inp}"
        yield _text_sample(user, out)


def iter_long_writer(seed: int):
    path = str(DATA_ROOT / "LongWriter-6k/long.jsonl")
    ds = _iter_jsonl(path, seed)
    for s in ds:
        msgs = s.get("messages")
        if msgs and len(msgs) >= 2:
            yield {"messages": msgs, "images": []}


def iter_science_qa(seed: int):
    ds = _iter_parquet(str(DATA_ROOT / "ScienceQA/data/train-*.parquet"), seed)
    for s in ds:
        img = _pil_from_field(s.get("image"))
        if img is None:
            # text-only ScienceQA rows are interesting but we focus on VL here
            continue
        q = s.get("question") or ""
        choices = s.get("choices") or []
        ans_i = s.get("answer")
        lec = s.get("lecture") or ""
        sol = s.get("solution") or ""
        if not q or ans_i is None:
            continue
        q_text = q
        if choices:
            q_text += "\n\n" + "\n".join(
                f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        parts = []
        if lec:
            parts.append(lec.strip())
        if sol:
            parts.append(sol.strip())
        if choices and 0 <= ans_i < len(choices):
            parts.append(f"Answer: {chr(65+ans_i)}. {choices[ans_i]}")
        a_text = "\n\n".join(parts) or "(no solution provided)"
        yield _vl_sample(q_text, a_text, images=[img])


def _pyarrow_row_stream(files: list[str], seed: int):
    """Stream rows from parquet files via pyarrow (no HF feature decoding).
    Yields plain dicts; image fields remain as {bytes, path} structs so the
    caller can decide how to decode (or skip if path is a dangling external)."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    order = list(files); rng.shuffle(order)
    for f in order:
        try:
            pf = pq.ParquetFile(f)
        except Exception:
            continue
        rg_idx = list(range(pf.num_row_groups))
        rng.shuffle(rg_idx)
        for gi in rg_idx:
            try:
                t = pf.read_row_group(gi)
            except Exception:
                continue
            rows = t.to_pylist()
            rng.shuffle(rows)
            for row in rows:
                yield row


def iter_cauldron(seed: int, skip_subsets: tuple[str, ...] = ("localized_narratives",)):
    """Stream the_cauldron uniformly across available subsets. Uses raw
    pyarrow so dangling image paths (e.g., clevr) yield None via
    `_pil_from_field` and we skip the sample instead of crashing."""
    root = DATA_ROOT / "the_cauldron"
    subsets = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in skip_subsets:
            continue
        files = sorted(str(f) for f in d.glob("*.parquet"))
        if files:
            subsets.append((d.name, files))
    if not subsets:
        raise FileNotFoundError(f"No cauldron subsets under {root}")
    rng = random.Random(seed + 9)
    iterators: dict[str, object] = {}
    while True:
        sub, files = rng.choice(subsets)
        if sub not in iterators:
            iterators[sub] = _pyarrow_row_stream(files, seed + hash(sub) % 10_000)
        try:
            row = next(iterators[sub])
        except StopIteration:
            # exhausted — restart with a new seed next time
            iterators.pop(sub, None)
            continue
        imgs_raw = row.get("images") or []
        texts = row.get("texts") or []
        pil_imgs = []
        for im in imgs_raw:
            try:
                p = _pil_from_field(im)
            except Exception:
                p = None
            if p is not None:
                pil_imgs.append(p)
        if not pil_imgs or not texts:
            continue
        for t in texts:
            u = (t.get("user") or "").strip()
            a = (t.get("assistant") or "").strip()
            if not u or not a:
                continue
            u = u.replace("<image>\n", "").replace("<image>", "").strip()
            yield _vl_sample(u, a, images=pil_imgs)


def iter_llava_onevision(seed: int,
                         skip_subsets: tuple[str, ...] = ()):
    """Stream LLaVA-OneVision-Data (3.2M VL samples, 93 subsets, bytes inline).
    Unified schema across subsets: {id, image{bytes,path}, conversations, data_source}.
    This dataset already subsumes cauldron + MathV360K + AI2D + ChartQA + Geometry3K
    etc. — use it as the single VL anchor for v3."""
    root = DATA_ROOT / "LLaVA-OneVision-Data"
    subsets = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in skip_subsets:
            continue
        files = sorted(str(f) for f in d.glob("*.parquet"))
        if files:
            subsets.append((d.name, files))
    if not subsets:
        raise FileNotFoundError(f"No LLaVA-OV subsets under {root}")
    rng = random.Random(seed + 13)
    iterators: dict[str, object] = {}
    while True:
        sub, files = rng.choice(subsets)
        if sub not in iterators:
            iterators[sub] = _pyarrow_row_stream(files, seed + hash(sub) % 10_000)
        try:
            row = next(iterators[sub])
        except StopIteration:
            iterators.pop(sub, None); continue
        img = None
        try:
            img = _pil_from_field(row.get("image"))
        except Exception:
            img = None
        convs = row.get("conversations") or []
        if img is None or not convs:
            continue
        msgs = _sharegpt_to_messages(convs, has_image=True)
        if msgs is None:
            continue
        yield {"messages": msgs, "images": [img]}


# ----------------------- registry ----------------------- #

_REGISTRY: dict[str, Callable[[int], Iterator[dict]]] = {
    "ultrachat":        iter_ultrachat,
    "llava_vsft":       iter_llava_vsft,
    "llava_onevision":  iter_llava_onevision,
    "numina_math":      iter_numina_math,
    "open_thoughts":    iter_open_thoughts,
    "openmath2":        iter_openmath2,
    "long_alpaca":      iter_long_alpaca,
    "long_writer":      iter_long_writer,
    "science_qa":       iter_science_qa,
    "cauldron":         iter_cauldron,
}

MODALITY: dict[str, str] = {
    "ultrachat": "text", "llava_vsft": "vlm", "llava_onevision": "vlm",
    "numina_math": "text", "open_thoughts": "text", "openmath2": "text",
    "long_alpaca": "text", "long_writer": "text",
    "science_qa": "vlm", "cauldron": "vlm",
}


# ----------------------- mixer + encode ----------------------- #

@dataclass
class MixSpec:
    """Weighted mix. Weights are auto-normalised."""
    weights: dict[str, float]  # name -> relative weight

    def names(self):
        return tuple(self.weights.keys())

    def normed(self):
        s = sum(self.weights.values()) or 1.0
        return [self.weights[n] / s for n in self.names()]


DEFAULT_V2_MIX = MixSpec(weights={
    # Reasoning-focused (no long-context sources; assume max_seq≤2048 so
    # long_alpaca / long_writer would mostly be filtered and waste the mixer).
    "ultrachat":     0.25,
    "numina_math":   0.20,
    "open_thoughts": 0.15,
    "openmath2":     0.05,
    "llava_vsft":    0.20,
    "science_qa":    0.05,
    "cauldron":      0.10,
})

V2_LONG_MIX = MixSpec(weights={
    # Variant for max_seq≥4096 training: reserves ~10% for long-context data.
    "ultrachat":     0.20,
    "numina_math":   0.15,
    "open_thoughts": 0.10,
    "openmath2":     0.05,
    "long_alpaca":   0.05,
    "long_writer":   0.05,
    "llava_vsft":    0.20,
    "science_qa":    0.05,
    "cauldron":      0.15,
})

V3_MIX = MixSpec(weights={
    # VL-heavy via LLaVA-OneVision (which internally subsumes cauldron +
    # MathV360K + AI2D + ChartQA + Geometry3K + …). Retains v2's text-math
    # reasoning signal at a reduced 20% budget.
    "llava_onevision": 0.60,
    "ultrachat":       0.20,
    "numina_math":     0.10,
    "open_thoughts":   0.10,
})

V3_VLONLY_MIX = MixSpec(weights={
    # Ablation: v3 without any text-math data. Tests H4 (math-CoT text as
    # adversary to diagram reasoning). If AI2D/MMStar recover here but
    # LAMBADA drops, H4 is confirmed.
    "llava_onevision": 0.80,
    "ultrachat":       0.20,
})


def parse_mix_string(spec: str) -> MixSpec:
    """`name1:w1,name2:w2,...`  absolute weights, auto-normalised.
    Accepts `equal` for uniform weights over all known sources."""
    if spec == "equal":
        return MixSpec(weights={n: 1.0 for n in _REGISTRY})
    if spec == "v2":
        return DEFAULT_V2_MIX
    if spec == "v2_long":
        return V2_LONG_MIX
    if spec == "v3":
        return V3_MIX
    if spec == "v3_vlonly":
        return V3_VLONLY_MIX
    if spec == "v1":
        return MixSpec(weights={"ultrachat": 1.0, "llava_vsft": 1.0})
    d: dict[str, float] = {}
    for tok in spec.split(","):
        if ":" not in tok:
            raise ValueError(f"bad mix token (want name:weight): {tok}")
        k, w = tok.split(":", 1)
        k = k.strip(); w = float(w.strip())
        if k not in _REGISTRY:
            raise ValueError(f"unknown source: {k}. known={list(_REGISTRY)}")
        d[k] = w
    if not d:
        raise ValueError("empty mix")
    return MixSpec(weights=d)


def build_mixed_stream(spec: MixSpec, seed: int = 0):
    """Yield (name, sample-dict) tuples forever. Each sample is in the
    unified format {messages, images}."""
    names = list(spec.names())
    weights = spec.normed()
    rng = random.Random(seed + 7)
    streams = {}
    for i, n in enumerate(names):
        streams[n] = iter(_REGISTRY[n](seed + i))
    while True:
        name = rng.choices(names, weights=weights, k=1)[0]
        it = streams[name]
        try:
            sample = next(it)
        except StopIteration:
            streams[name] = iter(_REGISTRY[name](seed + rng.randint(1, 10_000_000)))
            continue
        yield name, sample


def encode_sample(sample: dict, processor, max_seq: int,
                   compute_assistant_mask):
    """Apply chat template → processor → label mask. Returns
    (inputs, labels_1d) or None if the sample should be skipped
    (too long, no assistant tokens, tokenizer failure)."""
    import torch
    msgs = sample["messages"]
    imgs = sample.get("images") or []
    try:
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text],
                            images=imgs if imgs else None,
                            return_tensors="pt")
    except Exception:
        return None
    ids = inputs["input_ids"][0].tolist()
    if len(ids) > max_seq or len(ids) < 8:
        return None
    mask = compute_assistant_mask(ids)
    if sum(mask) == 0:
        return None
    labels = torch.tensor(
        [(t if m == 1 else -100) for t, m in zip(ids, mask)], dtype=torch.long,
    )
    return inputs, labels


# ----------------------- smoke inspect CLI ----------------------- #

def _smoke():
    """`python -m data_v2` prints 2 samples from each source to verify
    adapters work against current disk layout."""
    import sys, traceback
    for name, fn in _REGISTRY.items():
        print(f"\n=== {name} ({MODALITY[name]}) ===", flush=True)
        try:
            it = iter(fn(seed=0))
            for i in range(2):
                s = next(it)
                msgs = s["messages"]
                imgs = s.get("images") or []
                roles = " / ".join(m["role"] for m in msgs)
                first_text = next(
                    (m["content"] if isinstance(m["content"], str) else
                     (next((c["text"] for c in m["content"] if c.get("type") == "text"), ""))
                     for m in msgs if m["role"] == "user"), "")
                print(f"  sample {i}: n_msgs={len(msgs)} roles=({roles}) "
                      f"n_img={len(imgs)} first_user={first_text[:100]!r}")
        except Exception as e:
            print(f"  !! {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(limit=2)


if __name__ == "__main__":
    _smoke()
