#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

# ============================================================
# VIEW SELECTION (RE10K protocol) — same as reference
# ============================================================

N_CONTEXT = 2
N_TARGET = 3
MIN_FRAMES = N_CONTEXT + N_TARGET

def select_views(n_frames: int):
    """
    Deterministic evaluation protocol:
      context = [0, n-1]
      target  = 3 interior frames
    """
    if n_frames < MIN_FRAMES:
        return None

    context = [0, n_frames - 1]
    interior = list(range(1, n_frames - 1))

    if len(interior) < N_TARGET:
        return None

    step = len(interior) / (N_TARGET + 1)
    target = [interior[int((i + 1) * step)] for i in range(N_TARGET)]

    return context, target

# ============================================================
# TORCH DATASET SCAN — focused single-root version
# ============================================================

def build_torch_index(test_root: Path, strict: bool = False):
    """
    Scans test_root for *.torch files. Each torch file is expected to contain
    an iterable of scenes, where each scene is a dict with:
      - "key": unique scene id
      - "images": list/sequence of frames
    """
    index = {}

    if not test_root.exists():
        raise FileNotFoundError(f"Test folder not found: {test_root}")

    torch_files = sorted(test_root.glob("*.torch"))
    if not torch_files:
        raise FileNotFoundError(f"No .torch files found in: {test_root}")

    print(f"[torch] scanning {test_root} ({len(torch_files)} files)")

    for p in tqdm(torch_files):
        try:
            scenes = torch.load(p, map_location="cpu")
        except Exception as e:
            if strict:
                raise
            print(f"[warn] failed to load {p}: {e}")
            continue

        # scenes may be list/tuple; sometimes dict-like. We only handle iterables of scenes.
        try:
            iterator = iter(scenes)
        except TypeError:
            if strict:
                raise TypeError(f"{p} did not contain an iterable of scenes (type={type(scenes)})")
            print(f"[warn] {p} not iterable (type={type(scenes)}); skipping")
            continue

        for scene in iterator:
            # Validate minimal expected structure
            if not isinstance(scene, dict) or "key" not in scene or "images" not in scene:
                if strict:
                    raise KeyError(f"Scene in {p} missing required keys 'key'/'images': {scene.keys() if isinstance(scene, dict) else type(scene)}")
                continue

            key = scene["key"]
            images = scene["images"]

            try:
                n = len(images)
            except Exception:
                if strict:
                    raise
                continue

            views = select_views(n)
            if views is None:
                continue

            context, target = views
            index[key] = {"context": context, "target": target}

    return index

# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build evaluation_index_re10k.json from *.torch files in <root>/test and write it to <root>/."
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root containing train/ and test/ (only test/ is scanned).")
    ap.add_argument("--test_subdir", type=str, default="test",
                    help="Test subfolder name under root (default: test).")
    ap.add_argument("--out_name", type=str, default="evaluation_index_re10k.json",
                    help="Output JSON filename written next to train/test (default: evaluation_index_re10k.json).")
    ap.add_argument("--strict", action="store_true",
                    help="Fail fast on malformed files/scenes instead of skipping.")
    args = ap.parse_args()

    root = args.root
    test_root = root / args.test_subdir
    out_fp = root / args.out_name

    index = build_torch_index(test_root, strict=args.strict)

    with out_fp.open("w") as f:
        json.dump(index, f, indent=2)

    print("=" * 70)
    print(f"Torch-based index: {len(index)} scenes")
    print(f"  → wrote: {out_fp}")
    print("=" * 70)

if __name__ == "__main__":
    main()
