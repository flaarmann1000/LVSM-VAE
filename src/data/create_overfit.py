import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Source roots
PIXEL_SRC_ROOT  = Path("./data/data_processed/realestate10k/train")
LATENT_SRC_ROOT = Path("./data/data_processed/realestate10k_latent/train")

# Destination roots for different overfit variants
PIXEL_DST_ROOTS = {
    1: Path("./data/data_processed/realestate10k/overfit-1"),
    2: Path("./data/data_processed/realestate10k/overfit-2"),
    3: Path("./data/data_processed/realestate10k/overfit-3"),
}
LATENT_DST_ROOTS = {
    1: Path("./data/data_processed/realestate10k_latent/overfit-1"),
    2: Path("./data/data_processed/realestate10k_latent/overfit-2"),
    3: Path("./data/data_processed/realestate10k_latent/overfit-3"),
}

ASSETS_ROOT = Path("./assets")

FRAME_STRIDE = 15   # for overfit-1 filtering
MAX_FRAMES  = 5


# ---------- INDEX WRITER ----------

def write_overfit_index(version: int, scene_ids, assets_root: Path):
    """
    Writes:
        assets/overfitting_index_re10k-<version>.json

    Each scene_id gets exactly:
      "context": [0, 4]
      "target":  [1, 2, 3]
    """
    if not scene_ids:
        print(f"No scenes given for index v{version}, skipping.")
        return

    assets_root.mkdir(parents=True, exist_ok=True)
    fp = assets_root / f"overfitting_index_re10k-{version}.json"

    index_data = {
        scene_id: {
            "context": [0, 4],
            "target": [1, 2, 3],
        }
        for scene_id in scene_ids
    }

    with open(fp, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Saved index file (v{version}): {fp}")
    print(f"  scenes: {scene_ids}")


# ---------- SCENE PROCESSING ----------

def filter_scene(src_scene: Path, dst_scene: Path, latent: bool = False):
    """
    overfit-1 filtering:
    Copy every N-th image/.pt file and filter transforms.json accordingly.
    latent=True → expects .pt files instead of .png
    """
    dst_images = dst_scene / "images"
    dst_images.mkdir(parents=True, exist_ok=True)

    with open(src_scene / "transforms.json", "r") as f:
        meta = json.load(f)

    frames = meta["frames"]
    filtered_frames = frames[::FRAME_STRIDE]
    filtered_frames = filtered_frames[:MAX_FRAMES]

    print(f" Copying {len(filtered_frames)} frames → {dst_images}")

    for frame in tqdm(filtered_frames):
        src_file = src_scene / frame["file_path"]
        dst_file = dst_images / src_file.name
        shutil.copy2(src_file, dst_file)

        frame["file_path"] = f"images/{src_file.name}"

    meta["frames"] = filtered_frames

    with open(dst_scene / "transforms.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {dst_scene / 'transforms.json'}")


def copy_entire_scene(src_scene: Path, dst_scene: Path):
    """
    overfit-2 / overfit-3 copying:
    Copy the entire scene folder (all files, all frames) without modifying transforms.json.
    """
    print(f"Copying entire scene: {src_scene.name} → {dst_scene}")
    shutil.copytree(src_scene, dst_scene, dirs_exist_ok=True)
    print(f"Completed copy: {dst_scene}")


def process_scenes(src_root: Path, dst_root: Path, n_scenes: int, filtered: bool, latent: bool):
    """
    Process the first n_scenes from src_root into dst_root.

    - If filtered=True: applies overfit-1 logic (FRAME_STRIDE + MAX_FRAMES) on each scene.
    - If filtered=False: copies the entire scene directory.

    Returns:
        List of processed scene_ids (folder names).
    """
    scene_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    if not scene_dirs:
        print(f"No scenes found in: {src_root}")
        return []

    selected_scenes = scene_dirs[:n_scenes]
    processed_scene_ids = []

    for scene in selected_scenes:
        scene_id = scene.name
        processed_scene_ids.append(scene_id)
        print(f"Processing scene: {scene_id}")

        dst_scene = dst_root / scene_id
        dst_scene.mkdir(parents=True, exist_ok=True)

        if filtered:
            filter_scene(scene, dst_scene, latent=latent)
        else:
            copy_entire_scene(scene, dst_scene)

        print(f"Completed scene: {scene_id}\n")

    return processed_scene_ids


# ---------- MAIN ----------

if __name__ == "__main__":
    # Ensure destination roots exist
    for root in PIXEL_DST_ROOTS.values():
        root.mkdir(parents=True, exist_ok=True)
    for root in LATENT_DST_ROOTS.values():
        root.mkdir(parents=True, exist_ok=True)

    # ----- OVERFIT-1: original behavior, first scene only, filtered -----
    print("\n===== OVERFIT-1 (filtered first scene) =====")
    print("\n----- PIXEL DATASET -----")
    process_scenes(PIXEL_SRC_ROOT, PIXEL_DST_ROOTS[1], n_scenes=1, filtered=True, latent=False)

    print("\n----- LATENT DATASET -----")
    scenes_lat_1 = process_scenes(LATENT_SRC_ROOT, LATENT_DST_ROOTS[1], n_scenes=1, filtered=True, latent=True)

    # Index v1: first scene only
    if scenes_lat_1:
        write_overfit_index(1, [scenes_lat_1[0]], ASSETS_ROOT)

    # ----- OVERFIT-2: entire first scene -----
    print("\n===== OVERFIT-2 (entire first scene) =====")
    print("\n----- PIXEL DATASET -----")
    process_scenes(PIXEL_SRC_ROOT, PIXEL_DST_ROOTS[2], n_scenes=1, filtered=False, latent=False)

    print("\n----- LATENT DATASET -----")
    scenes_lat_2 = process_scenes(LATENT_SRC_ROOT, LATENT_DST_ROOTS[2], n_scenes=1, filtered=False, latent=True)

    # Index v2: identical to v1 (same single scene id)
    if scenes_lat_2:
        # scenes_lat_2[0] should be the same as scenes_lat_1[0] given sorted scenes
        write_overfit_index(2, [scenes_lat_2[0]], ASSETS_ROOT)

    # ----- OVERFIT-3: first 3 scenes (entire scenes) -----
    print("\n===== OVERFIT-3 (first 3 full scenes) =====")
    print("\n----- PIXEL DATASET -----")
    process_scenes(PIXEL_SRC_ROOT, PIXEL_DST_ROOTS[3], n_scenes=3, filtered=False, latent=False)

    print("\n----- LATENT DATASET -----")
    scenes_lat_3 = process_scenes(LATENT_SRC_ROOT, LATENT_DST_ROOTS[3], n_scenes=3, filtered=False, latent=True)

    # Index v3: all three scene ids
    if scenes_lat_3:
        write_overfit_index(3, scenes_lat_3, ASSETS_ROOT)

    print("\nDone! Overfit-1 / -2 / -3 datasets and index files created.")
