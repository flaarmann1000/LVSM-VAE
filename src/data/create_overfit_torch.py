import json
import torch
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path("./data/re10k_subset")
ASSETS_ROOT = Path("./assets")

TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT  = DATA_ROOT / "test"

OUT_TRAIN_ROOT = DATA_ROOT
OUT_TEST_ROOT  = DATA_ROOT

# Overfit-1 params
FRAME_STRIDE = 15
MAX_FRAMES = 5

# Overfit-4 params (must match original script)
TEST_SEQ_LEN = 5
N_TRAIN_SCENES_V4 = 10
N_SEEN_TEST_SCENES_V4 = 2
N_UNSEEN_TEST_SCENES_V4 = 2


# ============================================================
# LOADING UTILITIES
# ============================================================

def load_all_scenes(root: Path):
    scenes = []
    for f in sorted(root.glob("*.torch")):
        scenes.extend(torch.load(f))
    return scenes


def assert_unique_keys(scenes):
    keys = [s["key"] for s in scenes]
    assert len(keys) == len(set(keys)), "Duplicate scene keys detected"


# ============================================================
# INDEX WRITER (TORCH VERSION)
# ============================================================

def write_overfit_index_torch(version: int, scenes):
    """
    Writes:
        assets/overfitting_index_re10k-t<version>.json
    """
    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    fp = ASSETS_ROOT / f"overfitting_index_re10k-t{version}.json"

    data = {
        s["key"]: {
            "context": [0, 4],
            "target": [1, 2, 3],
        }
        for s in scenes
    }

    with open(fp, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote index file → {fp}")
    print("  scenes:", list(data.keys()))


# ============================================================
# FRAME HELPERS
# ============================================================

def filter_scene_overfit1(scene):
    idxs = list(range(0, len(scene["images"]), FRAME_STRIDE))[:MAX_FRAMES]
    return {
        **scene,
        "timestamps": scene["timestamps"][idxs],
        "cameras": scene["cameras"][idxs],
        "images": [scene["images"][i] for i in idxs],
    }


def take_first_frames(scene, n):
    idxs = list(range(min(n, len(scene["images"]))))
    return {
        **scene,
        "timestamps": scene["timestamps"][idxs],
        "cameras": scene["cameras"][idxs],
        "images": [scene["images"][i] for i in idxs],
    }


def take_last_frames(scene, n):
    start = max(0, len(scene["images"]) - n)
    idxs = list(range(start, len(scene["images"])))
    return {
        **scene,
        "timestamps": scene["timestamps"][idxs],
        "cameras": scene["cameras"][idxs],
        "images": [scene["images"][i] for i in idxs],
    }


# ============================================================
# OVERFIT 1 / 2 / 3
# ============================================================

def build_overfit_simple(
    src_root: Path,
    out_train: Path,
    out_test: Path,
    n_scenes: int,
    filtered: bool,
    version: int,
):
    scenes = load_all_scenes(src_root)
    assert_unique_keys(scenes)

    selected = scenes[:n_scenes]
    if filtered:
        selected = [filter_scene_overfit1(s) for s in selected]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    torch.save(selected, out_train)
    torch.save(selected, out_test)

    print(f"Saved {len(selected)} scenes → {out_train}")
    print(f"Saved {len(selected)} scenes → {out_test}")

    # index uses TEST scenes (same as train for v1–3)
    write_overfit_index_torch(version, selected)


# ============================================================
# OVERFIT-4
# ============================================================

def build_overfit_4(
    src_root: Path,
    out_train: Path,
    out_test: Path,
):
    scenes = load_all_scenes(src_root)
    assert_unique_keys(scenes)

    train_ids = scenes[:N_TRAIN_SCENES_V4]
    seen_ids  = train_ids[:N_SEEN_TEST_SCENES_V4]
    unseen_ids = scenes[
        N_TRAIN_SCENES_V4 :
        N_TRAIN_SCENES_V4 + N_UNSEEN_TEST_SCENES_V4
    ]

    train_out = []
    test_out  = []

    for s in train_ids:
        if s in seen_ids:
            train_out.append(
                take_first_frames(s, len(s["images"]) - TEST_SEQ_LEN)
            )
            test_out.append(
                take_last_frames(s, TEST_SEQ_LEN)
            )
        else:
            train_out.append(s)

    for s in unseen_ids:
        test_out.append(take_first_frames(s, TEST_SEQ_LEN))

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    torch.save(train_out, out_train)
    torch.save(test_out, out_test)

    print(f"Saved {len(train_out)} train scenes → {out_train}")
    print(f"Saved {len(test_out)} test scenes  → {out_test}")

    # index is written ONLY over test scenes (same as original)
    write_overfit_index_torch(4, test_out)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n===== OVERFIT-1 =====")
    build_overfit_simple(
        TRAIN_ROOT,
        OUT_TRAIN_ROOT / "train-overfit-1/00000.torch",
        OUT_TEST_ROOT  / "test-overfit-1/00000.torch",
        n_scenes=1,
        filtered=True,
        version=1,
    )

    print("\n===== OVERFIT-2 =====")
    build_overfit_simple(
        TRAIN_ROOT,
        OUT_TRAIN_ROOT / "train-overfit-2/00000.torch",
        OUT_TEST_ROOT  / "test-overfit-2/00000.torch",
        n_scenes=1,
        filtered=False,
        version=2,
    )

    print("\n===== OVERFIT-3 =====")
    build_overfit_simple(
        TRAIN_ROOT,
        OUT_TRAIN_ROOT / "train-overfit-3/00000.torch",
        OUT_TEST_ROOT  / "test-overfit-3/00000.torch",
        n_scenes=3,
        filtered=False,
        version=3,
    )

    print("\n===== OVERFIT-4 =====")
    build_overfit_4(
        TRAIN_ROOT,
        OUT_TRAIN_ROOT / "train-overfit-4/00000.torch",
        OUT_TEST_ROOT  / "test-overfit-4/00000.torch",
    )

    print("\nAll torch overfit datasets and indices built successfully!\n")
