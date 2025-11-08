import os
from pathlib import Path
from tqdm import tqdm
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

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

# Input / output folders
src_dir = Path("./scratch/partial_datasets/realestate10k/train/0c6c060f599465ed/images")
dst_dir = Path("./scratch/partial_datasets/realestate10k_latent/train/0c6c060f599465ed/images")
dst_dir.mkdir(parents=True, exist_ok=True)

# Iterate over all PNG files
for img_path in tqdm(sorted(src_dir.glob("*.png"))):
    # Load and preprocess
    print(f"encoding {img_path}")
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32)
    img_tensor = img_tensor / 255.0 * 2 - 1
    img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(device, dtype=DTYPE)

    # Encode
    with torch.no_grad():
        out = vae.encode(img_tensor)
        z = out.latent_dist.mean  # deterministic latent

    # Save latent as .pt
    save_path = dst_dir / (img_path.stem + ".pt")
    torch.save(z.cpu(), save_path)

print("All latents saved to:", dst_dir)
