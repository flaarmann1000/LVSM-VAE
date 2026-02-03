import json
from pathlib import Path
import torch
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

# Torch-based datasets
TORCH_ROOTS = [
    Path("./data/re10k_subset/test"),
    Path("./data/re10k_subset_latent/test"),
]

# Folder-based dataset
RE10K_ROOT = Path("./data/data_processed/realestate10k/test")

OUT_FOLDER_INDEX = Path("./assets/evaluation_index_re10k_subset.json")
OUT_TORCH_INDEX  = Path("./assets/evaluation_index_re10k_subset_t.json")

N_CONTEXT = 2
N_TARGET = 3
MIN_FRAMES = N_CONTEXT + N_TARGET

# ============================================================
# VIEW SELECTION (RE10K protocol)
# ============================================================

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
# TORCH DATASET SCAN
# ============================================================

def build_torch_index():
    index = {}

    for root in TORCH_ROOTS:
        if not root.exists():
            continue

        torch_files = sorted(root.glob("*.torch"))
        if not torch_files:
            continue

        print(f"[torch] scanning {root}")

        for p in tqdm(torch_files):
            scenes = torch.load(p)
            for scene in scenes:
                key = scene["key"]
                n = len(scene["images"])

                views = select_views(n)
                if views is None:
                    continue

                context, target = views
                index[key] = {
                    "context": context,
                    "target": target,
                }

    return index

# ============================================================
# FOLDER DATASET SCAN
# ============================================================

def build_folder_index():
    index = {}

    if not RE10K_ROOT.exists():
        return index

    print(f"[folder] scanning {RE10K_ROOT}")

    for scene_dir in tqdm(sorted(RE10K_ROOT.iterdir())):
        if not scene_dir.is_dir():
            continue

        transforms_fp = scene_dir / "transforms.json"
        if not transforms_fp.exists():
            continue

        try:
            with open(transforms_fp) as f:
                meta = json.load(f)
        except Exception:
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
    folder_index = build_folder_index()
    torch_index  = build_torch_index()

    OUT_FOLDER_INDEX.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FOLDER_INDEX, "w") as f:
        json.dump(folder_index, f, indent=2)

    with open(OUT_TORCH_INDEX, "w") as f:
        json.dump(torch_index, f, indent=2)

    print("=" * 70)
    print(f"Folder-based index : {len(folder_index)} scenes")
    print(f"  → {OUT_FOLDER_INDEX}")
    print(f"Torch-based index  : {len(torch_index)} scenes")
    print(f"  → {OUT_TORCH_INDEX}")
    print("=" * 70)


if __name__ == "__main__":
    main()
