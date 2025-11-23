import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

PIXEL_SRC_ROOT  = Path("./data/data_processed/realestate10k/train")
LATENT_SRC_ROOT = Path("./data/data_processed/realestate10k_latent/train")

PIXEL_DST_ROOT  = Path("./data/data_processed/realestate10k/overfit")
LATENT_DST_ROOT = Path("./data/data_processed/realestate10k_latent/overfit")

FRAME_STRIDE = 15   # keep every 15th image

MAX_FRAMES = 5


def filter_scene(src_scene: Path, dst_scene: Path, latent: bool = False):
    """
    Copy every N-th image/.pt file and filter transforms.json accordingly.
    latent=True → expects .pt files instead of .png
    """
    dst_images = dst_scene / "images"
    dst_images.mkdir(parents=True, exist_ok=True)
    
    with open(src_scene / "transforms.json", "r") as f:
        meta = json.load(f)

    frames = meta["frames"]
    filtered_frames = frames[::FRAME_STRIDE]  # keep every 15th frame
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


def process_first_scene(src_root: Path, dst_root: Path, latent=False):
    """Find the first scene alphabetically and create the overfit version."""
    scene_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])    
    if not scene_dirs:
        print(f"No scenes found in: {src_root}")
        return

    first_scene = scene_dirs[0]
    print(f"Processing scene: {first_scene.name}")

    dst_scene = dst_root / first_scene.name
    dst_scene.mkdir(parents=True, exist_ok=True)

    filter_scene(first_scene, dst_scene, latent=latent)

    print(f"Completed scene: {first_scene.name}\n")


if __name__ == "__main__":
    PIXEL_DST_ROOT.mkdir(parents=True, exist_ok=True)
    LATENT_DST_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n----- PIXEL DATASET -----")
    process_first_scene(PIXEL_SRC_ROOT, PIXEL_DST_ROOT, latent=False)

    print("\n----- LATENT DATASET -----")
    process_first_scene(LATENT_SRC_ROOT, LATENT_DST_ROOT, latent=True)

    print("\nDone! Overfit dataset created.")
