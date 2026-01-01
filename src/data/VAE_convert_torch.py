import torch
from pathlib import Path
from typing import Tuple, List
from diffusers import AutoencoderKL
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

SRC_ROOT = Path("./data/re10k")
DST_ROOT = Path("/mnt/hdd/data/re10k_latent_half")

PATCH_SIZE = 256
VAE_SCALE = 8
DTYPE = torch.float32

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# VAE
# ============================================================

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="vae",
    torch_dtype=DTYPE,
).to(device)
vae.eval()

# ============================================================
# IMAGE + INTRINSIC TRANSFORM
# ============================================================

def resize_crop_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exactly the same logic as your previous script.
    """
    h, w = image.shape[:2]
    scale = patch_size / min(h, w)

    new_w, new_h = w * scale, h * scale
    crop_x = (new_w - patch_size) / 2
    crop_y = (new_h - patch_size) / 2

    M = np.array(
        [[scale, 0, -crop_x],
         [0, scale, -crop_y]],
        dtype=np.float32
    )

    is_downsampling = min(h, w) > patch_size
    interp = cv2.INTER_AREA if is_downsampling else cv2.INTER_CUBIC

    out = cv2.warpAffine(
        image, M, (patch_size, patch_size), flags=interp
    )

    K_new = K.copy()
    K_new[:2, :] *= scale
    K_new[0, 2] -= crop_x
    K_new[1, 2] -= crop_y

    return out, K_new


# ============================================================
# SCENE CONVERSION
# ============================================================

@torch.no_grad()
def convert_scene(scene: dict) -> dict:
    """
    Converts one PixelSplat-style scene dict to latent version.
    """
    new_scene = dict(scene)
    new_images: List[torch.Tensor] = []

    cameras = scene["cameras"]
    images = scene["images"]

    for i, img_bytes in enumerate(images):
        # Decode JPEG bytes
        img = Image.open(BytesIO(img_bytes.numpy().tobytes())).convert("RGB")
        img_np = np.array(img)

        # Intrinsics
        fx, fy, cx, cy = cameras[i, :4].cpu().numpy()
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float32)

        # Resize + crop
        img_proc, K_new = resize_crop_with_subpixel_accuracy(
            img_np, K, PATCH_SIZE
        )

        # Update camera intrinsics in-place
        cameras[i, 0] = float(K_new[0, 0])
        cameras[i, 1] = float(K_new[1, 1])
        cameras[i, 2] = float(K_new[0, 2])
        cameras[i, 3] = float(K_new[1, 2])

        # To tensor [-1, 1]
        img_tensor = (
            torch.from_numpy(img_proc)
            .float()
            .to(device)
            / 255.0 * 2.0 - 1.0
        )
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Encode
        latent = vae.encode(img_tensor).latent_dist.mean
        new_images.append(latent.cpu())

    new_scene["images"] = new_images
    new_scene["cameras"] = cameras

    return new_scene


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    all_subdirs = [
        p for p in SRC_ROOT.iterdir()
        if p.is_dir()
    ]

    print(f"Found {len(all_subdirs)} dataset folders")

    for subdir in all_subdirs:
        dst_subdir = DST_ROOT / subdir.name
        dst_subdir.mkdir(parents=True, exist_ok=True)

        torch_files = sorted(subdir.glob("*.torch"))
        print(f"\n{subdir.name}: {len(torch_files)} files")
        l_torch_files = len(torch_files)

        for src_fp in torch_files[:l_torch_files//2]:
            dst_fp = dst_subdir / src_fp.name
            print(f"  → {dst_fp}")

            scenes = torch.load(src_fp)
            new_scenes = []

            for scene in tqdm(scenes, leave=False):
                new_scenes.append(convert_scene(scene))

            torch.save(new_scenes, dst_fp)

    print("\nLatent re10k_subset dataset created successfully.")


if __name__ == "__main__":
    main()
