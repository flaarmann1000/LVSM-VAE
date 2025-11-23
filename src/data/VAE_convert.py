import os
from pathlib import Path
from tqdm import tqdm
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import torch.nn.functional as F
import json
from typing import Tuple
import cv2

PATCH_SIZE = 256
VAE_SCALE = 8

# Input / output folders
# mode = "train"
mode = "test"

root_src = Path(f"./data/data_processed/realestate10k/{mode}")
root_dst = Path(f"./data/data_processed/realestate10k_latent/{mode}")
root_dst.mkdir(parents=True, exist_ok=True)    

def resize_crop_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize and crop the image to have the smallest side equal to `patch_size`,
    while maintaining sub-pixel accuracy using a single warpAffine transformation.

    Args:
        image (np.ndarray): Input image.
        K (np.ndarray): Camera intrinsic matrix.
        patch_size (int): Target size of the smaller dimension.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resized and cropped image, updated intrinsic matrix.
    """
    h, w = image.shape[:2]
    scale = patch_size / min(h, w)

    # Compute the affine transformation matrix combining scaling and cropping
    new_w, new_h = w * scale, h * scale
    crop_x = (new_w - patch_size) / 2
    crop_y = (new_h - patch_size) / 2

    M = np.array([[scale, 0, -crop_x], [0, scale, -crop_y]], dtype=np.float32)

    # Apply affine transformation with sub-pixel accuracy
    is_downsampling = min(h, w) > patch_size
    interpolation = cv2.INTER_AREA if is_downsampling else cv2.INTER_CUBIC
    cropped_resized_image = cv2.warpAffine(
        image, M, (patch_size, patch_size), flags=interpolation
    )

    # Update intrinsic matrix K
    K_scaled = K.copy()
    K_scaled[:2, :] *= scale
    K_scaled[0, 2] -= crop_x
    K_scaled[1, 2] -= crop_y

    return cropped_resized_image, K_scaled

# Configuration
DTYPE = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load VAE
vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="vae",
    torch_dtype=DTYPE
).to(device)
vae.eval()



for seq_dir in sorted(root_src.iterdir()):
    if not seq_dir.is_dir():
        continue

    print(f"Processing sequence: {seq_dir.name}")

    src_dir = seq_dir
    dst_dir = root_dst / seq_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)    
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)

    src_img_dir = src_dir / "images"
    json_path = src_dir / "transforms.json"

    with open(json_path, "r") as f:
        meta_info = json.load(f)

    K = np.array([
        [meta_info["fl_x"], 0, meta_info["cx"]],
        [0, meta_info["fl_y"], meta_info["cy"]],
        [0, 0, 1],
    ], dtype=np.float32)

    # Iterate over all PNG files
    for img_path in tqdm(sorted(src_img_dir.glob("*.png"))):
        # Load and preprocess
        # print(f"encoding {img_path}")
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32)
        img_tensor = img_tensor / 255.0 * 2 - 1    
        # H W C
        
        img_tensor, K_scaled = resize_crop_with_subpixel_accuracy(img_tensor.numpy(), K, patch_size=PATCH_SIZE)    
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        # Encode
        with torch.no_grad():
            out = vae.encode(img_tensor)
            z = out.latent_dist.mean  # deterministic latent

        # Save latent as .pt
        save_path = dst_dir / "images" / (img_path.stem + ".pt")
        torch.save(z.cpu(), save_path)



    fx_new, fy_new = K_scaled[0, 0], K_scaled[1, 1]
    cx_new, cy_new = K_scaled[0, 2], K_scaled[1, 2]
    w_new = PATCH_SIZE // VAE_SCALE
    h_new = PATCH_SIZE // VAE_SCALE

    meta_info_latent = meta_info.copy()
    meta_info_latent.update({
        "fl_x": float(fx_new),
        "fl_y": float(fy_new),
        "cx": float(cx_new),
        "cy": float(cy_new),
        "w": int(w_new),
        "h": int(h_new),
    })

    # rename files from .png to .pt
    text = json.dumps(meta_info_latent)
    text = text.replace(".png", ".pt")
    data_new = json.loads(text)


    latent_json_path = dst_dir / "transforms.json"
    with open(latent_json_path, "w") as f:
        json.dump(data_new, f, indent=2)

    print(f"Finished: {seq_dir.name}")

