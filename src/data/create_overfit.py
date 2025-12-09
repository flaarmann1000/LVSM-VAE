import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

PIXEL_SRC_ROOT  = Path("./data/data_processed/realestate10k/train")
LATENT_SRC_ROOT = Path("./data/data_processed/realestate10k_latent/train")

PIXEL_TRAIN_ROOTS = {
    1: Path("./data/data_processed/realestate10k/train-overfit-1"),
    2: Path("./data/data_processed/realestate10k/train-overfit-2"),
    3: Path("./data/data_processed/realestate10k/train-overfit-3"),
    4: Path("./data/data_processed/realestate10k/train-overfit-4"),
}
PIXEL_TEST_ROOTS = {
    1: Path("./data/data_processed/realestate10k/test-overfit-1"),
    2: Path("./data/data_processed/realestate10k/test-overfit-2"),
    3: Path("./data/data_processed/realestate10k/test-overfit-3"),
    4: Path("./data/data_processed/realestate10k/test-overfit-4"),
}

LATENT_TRAIN_ROOTS = {
    1: Path("./data/data_processed/realestate10k_latent/train-overfit-1"),
    2: Path("./data/data_processed/realestate10k_latent/train-overfit-2"),
    3: Path("./data/data_processed/realestate10k_latent/train-overfit-3"),
    4: Path("./data/data_processed/realestate10k_latent/train-overfit-4"),
}
LATENT_TEST_ROOTS = {
    1: Path("./data/data_processed/realestate10k_latent/test-overfit-1"),
    2: Path("./data/data_processed/realestate10k_latent/test-overfit-2"),
    3: Path("./data/data_processed/realestate10k_latent/test-overfit-3"),
    4: Path("./data/data_processed/realestate10k_latent/test-overfit-4"),
}

ASSETS_ROOT = Path("./assets")

# Overfit-1 params
FRAME_STRIDE = 15
MAX_FRAMES  = 5

# Overfit-4 params
TEST_SEQ_LEN = 5
N_TRAIN_SCENES_V4 = 10
N_SEEN_TEST_SCENES_V4 = 2
N_UNSEEN_TEST_SCENES_V4 = 2


# ============================================================
# INDEX WRITER
# ============================================================

def write_overfit_index(version: int, scene_ids, assets_root: Path):
    """
    Writes:

        assets/overfitting_index_re10k-<version>.json

    For v1–3: scene_ids are the (latent) train/test scenes.
    For v4:   scene_ids are the latent TEST scenes (seen + unseen).
    """
    fp = assets_root / f"overfitting_index_re10k-{version}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)

    data = {
        sid: {
            "context": [0, 4],
            "target": [1, 2, 3]
        }
        for sid in scene_ids
    }

    with open(fp, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote index file v{version} → {fp}")
    print("  scenes:", scene_ids)


# ============================================================
# FRAME COPY HELPERS
# ============================================================

def copy_frames(src_scene: Path, dst_scene: Path, frame_indices=None):
    """
    Copies selected frames, or all frames when frame_indices=None.
    Writes a new transforms.json.
    """
    with open(src_scene / "transforms.json") as f:
        meta = json.load(f)

    frames = meta["frames"]
    dst_images = dst_scene / "images"
    dst_images.mkdir(parents=True, exist_ok=True)

    new_frames = []

    if frame_indices is None:
        selected = enumerate(frames)
    else:
        selected = [(i, frames[i]) for i in frame_indices]

    for _, frame in selected:
        frame = dict(frame)

        src_file = src_scene / frame["file_path"]
        dst_file = dst_images / src_file.name
        shutil.copy2(src_file, dst_file)

        frame["file_path"] = f"./images/{src_file.name}"
        new_frames.append(frame)

    meta["frames"] = new_frames

    with open(dst_scene / "transforms.json", "w") as f:
        json.dump(meta, f, indent=2)


def filter_scene_overfit1(src_scene: Path, dst_scene: Path):
    """overfit-1: every 15th frame, max 5 frames."""
    with open(src_scene / "transforms.json") as f:
        meta = json.load(f)

    frames = meta["frames"]
    chosen = frames[::FRAME_STRIDE][:MAX_FRAMES]
    idxs = [frames.index(f) for f in chosen]

    copy_frames(src_scene, dst_scene, idxs)


def copy_entire_scene(src_scene: Path, dst_scene: Path):
    copy_frames(src_scene, dst_scene, None)


# ============================================================
# OVERFIT 1 / 2 / 3
# ============================================================

def build_overfit_simple(src_root: Path, train_root: Path, test_root: Path,
                         n_scenes: int, filtered: bool):
    """
    Build train-overfit-X/ and test-overfit-X/ for versions 1–3.

    For v1–3 we mimic the original behavior:
    - same scenes & same content are used for both train and test.
    """
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    selected = scene_dirs[:n_scenes]
    scene_ids = []

    for scene in selected:
        sid = scene.name
        scene_ids.append(sid)
        print(f"[simple] scene: {sid}")

        # train copy
        dst_train = train_root / sid
        dst_train.mkdir(parents=True, exist_ok=True)

        # test copy (same content for v1–3)
        dst_test = test_root / sid
        dst_test.mkdir(parents=True, exist_ok=True)

        if filtered:
            filter_scene_overfit1(scene, dst_train)
            filter_scene_overfit1(scene, dst_test)
        else:
            copy_entire_scene(scene, dst_train)
            copy_entire_scene(scene, dst_test)

    # For v1–3, test_ids == train_ids == scene_ids
    return scene_ids, scene_ids  # (test_ids, train_ids)


# ============================================================
# OVERFIT-4 BUILD
# ============================================================

def build_overfit_4(src_root: Path, train_root: Path, test_root: Path,
                    all_scene_ids):
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    train_ids = all_scene_ids[:N_TRAIN_SCENES_V4]
    seen_ids  = train_ids[:N_SEEN_TEST_SCENES_V4]
    unseen_ids = all_scene_ids[
        N_TRAIN_SCENES_V4 : N_TRAIN_SCENES_V4 + N_UNSEEN_TEST_SCENES_V4
    ]

    # --------------------
    # TRAIN
    # --------------------
    for sid in train_ids:
        src = src_root / sid
        dst = train_root / sid
        dst.mkdir(parents=True, exist_ok=True)

        with open(src / "transforms.json") as f:
            meta = json.load(f)
        n = len(meta["frames"])

        if sid in seen_ids:
            holdout = list(range(max(0, n - TEST_SEQ_LEN), n))
            keep = [i for i in range(n) if i not in holdout]
        else:
            keep = list(range(n))

        copy_frames(src, dst, keep)

    # --------------------
    # TEST (seen)
    # --------------------
    for sid in seen_ids:
        src = src_root / sid
        dst = test_root / sid
        dst.mkdir(parents=True, exist_ok=True)

        with open(src / "transforms.json") as f:
            meta = json.load(f)
        n = len(meta["frames"])

        test_idx = list(range(max(0, n - TEST_SEQ_LEN), n))
        copy_frames(src, dst, test_idx)

    # --------------------
    # TEST (unseen)
    # --------------------
    for sid in unseen_ids:
        src = src_root / sid
        dst = test_root / sid
        dst.mkdir(parents=True, exist_ok=True)

        with open(src / "transforms.json") as f:
            meta = json.load(f)
        n = len(meta["frames"])

        test_idx = list(range(min(TEST_SEQ_LEN, n)))
        copy_frames(src, dst, test_idx)

    return seen_ids + unseen_ids, train_ids  # (test_ids, train_ids)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ------------------
    # OVERFIT-1
    # ------------------
    print("\n===== OVERFIT-1 =====")
    test_ids_1_pixel, train_ids_1_pixel = build_overfit_simple(
        PIXEL_SRC_ROOT, PIXEL_TRAIN_ROOTS[1], PIXEL_TEST_ROOTS[1],
        n_scenes=1, filtered=True
    )
    test_ids_1_latent, train_ids_1_latent = build_overfit_simple(
        LATENT_SRC_ROOT, LATENT_TRAIN_ROOTS[1], LATENT_TEST_ROOTS[1],
        n_scenes=1, filtered=True
    )

    # original behavior: index contains *first latent scene only*
    write_overfit_index(1, train_ids_1_latent, ASSETS_ROOT)


    # ------------------
    # OVERFIT-2
    # ------------------
    print("\n===== OVERFIT-2 =====")
    test_ids_2_pixel, train_ids_2_pixel = build_overfit_simple(
        PIXEL_SRC_ROOT, PIXEL_TRAIN_ROOTS[2], PIXEL_TEST_ROOTS[2],
        n_scenes=1, filtered=False
    )
    test_ids_2_latent, train_ids_2_latent = build_overfit_simple(
        LATENT_SRC_ROOT, LATENT_TRAIN_ROOTS[2], LATENT_TEST_ROOTS[2],
        n_scenes=1, filtered=False
    )

    # original behavior: index identical to v1 (same scene id)
    write_overfit_index(2, train_ids_2_latent, ASSETS_ROOT)


    # ------------------
    # OVERFIT-3
    # ------------------
    print("\n===== OVERFIT-3 =====")
    test_ids_3_pixel, train_ids_3_pixel = build_overfit_simple(
        PIXEL_SRC_ROOT, PIXEL_TRAIN_ROOTS[3], PIXEL_TEST_ROOTS[3],
        n_scenes=3, filtered=False
    )
    test_ids_3_latent, train_ids_3_latent = build_overfit_simple(
        LATENT_SRC_ROOT, LATENT_TRAIN_ROOTS[3], LATENT_TEST_ROOTS[3],
        n_scenes=3, filtered=False
    )

    # original behavior: index contains all three latent scene ids
    write_overfit_index(3, train_ids_3_latent, ASSETS_ROOT)


    # ------------------
    # OVERFIT-4
    # ------------------
    print("\n===== OVERFIT-4 =====")

    # canonical ordering from latent dataset
    all_latent_ids = sorted([d.name for d in LATENT_SRC_ROOT.iterdir() if d.is_dir()])

    # if you truly want pixel & latent to use exactly the same scene IDs,
    # we reuse all_latent_ids for both (assuming sets match on disk)
    test_ids_pixel_4, train_ids_pixel_4 = build_overfit_4(
        PIXEL_SRC_ROOT,
        PIXEL_TRAIN_ROOTS[4],
        PIXEL_TEST_ROOTS[4],
        all_latent_ids
    )
    test_ids_latent_4, train_ids_latent_4 = build_overfit_4(
        LATENT_SRC_ROOT,
        LATENT_TRAIN_ROOTS[4],
        LATENT_TEST_ROOTS[4],
        all_latent_ids
    )

    # write index for test scenes only (latent)
    write_overfit_index(4, test_ids_latent_4, ASSETS_ROOT)

    print("\nAll overfit datasets built successfully!\n")
