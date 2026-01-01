# Creates an eval index in dataset folder
# to be used with the -data flag in main.py

# Usage example
# python src/data/build_eval_index.py --root /mnt/hdd/data/re10k_latent_half


#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# ============================================================
# VIEW SELECTION (RE10K protocol)
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
# INDEX BUILD
# ============================================================

def build_folder_index(test_root: Path):
    index = {}

    if not test_root.exists():
        raise FileNotFoundError(f"Test folder not found: {test_root}")

    print(f"[folder] scanning {test_root}")

    for scene_dir in tqdm(sorted(test_root.iterdir())):
        if not scene_dir.is_dir():
            continue

        transforms_fp = scene_dir / "transforms.json"
        if not transforms_fp.exists():
            continue

        try:
            with transforms_fp.open("r") as f:
                meta = json.load(f)
        except Exception:
            # If a transforms.json is malformed, skip it (matches your original spirit)
            continue

        frames = meta.get("frames", [])
        n = len(frames)

        views = select_views(n)
        if views is None:
            continue

        context, target = views
        index[scene_dir.name] = {
            "context": context,
            "target": target,
        }

    return index

# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Create evaluation_index_re10k.json next to train/test for a given dataset root."
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing train/ and test/ subfolders (only test/ is scanned).",
    )
    ap.add_argument(
        "--test_subdir",
        type=str,
        default="test",
        help="Name of the test subfolder under root (default: test).",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="evaluation_index_re10k.json",
        help="Output json filename to place at root/ (default: evaluation_index_re10k.json).",
    )
    args = ap.parse_args()

    root = args.root
    test_root = root / args.test_subdir

    index = build_folder_index(test_root)

    out_fp = root / args.out_name
    with out_fp.open("w") as f:
        json.dump(index, f, indent=2)

    print("=" * 70)
    print(f"Index scenes: {len(index)}")
    print(f"  → wrote: {out_fp}")
    print("=" * 70)

if __name__ == "__main__":
    main()
