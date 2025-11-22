import os
import gc
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers.models import AutoencoderKL

# ---------- CONFIG ----------
INPUT_DIR = "data/single"
OUTPUT_DIR = "data/latent"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
DOWNSAMPLE = 8                  # VAE downsampling factor (usually 8)
STORE_DTYPE = torch.float16     # compact dtype for saved latents
BATCH_SIZE = 1                  # number of images processed at once
USE_COMPILE = True              # set False if using PyTorch < 2.1
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading VAE on {DEVICE} ({MODEL_NAME})...")
vae = AutoencoderKL.from_pretrained(
    MODEL_NAME,
    subfolder="vae",
    torch_dtype=torch.float16
).to(DEVICE)
vae.eval()

# Optional: compile for fused kernels (can improve speed 10–30%)
if USE_COMPILE and hasattr(torch, "compile"):
    vae = torch.compile(vae, mode="max-autotune")
    print("✅ Compiled VAE with torch.compile().")

print("✅ VAE loaded successfully.\n")


# ---------- UTILS ----------

def to_multiple_of(x, k):
    return max(k, (x // k) * k)


def preprocess_image(img_bytes):
    """Decode bytes → normalized tensor in [-1,1], shape (1,3,H,W)."""
    img = Image.open(io.BytesIO(img_bytes.numpy())).convert("RGB")
    w, h = img.size
    nw, nh = to_multiple_of(w, DOWNSAMPLE), to_multiple_of(h, DOWNSAMPLE)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    tensor = tensor * 2 - 1
    return tensor


@torch.inference_mode()
def encode_batch(batch_tensors):
    """Run VAE encoder on a batch."""
    batch_tensors = batch_tensors.to(device=DEVICE, dtype=torch.float16, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        encoded = vae.encode(batch_tensors).latent_dist.mean
    return encoded.cpu().to(dtype=STORE_DTYPE)


# ---------- MAIN LOOP ----------

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".torch")]
print(f"Found {len(files)} .torch files in {INPUT_DIR}\n")

for file_idx, fname in enumerate(files, 1):
    fpath = os.path.join(INPUT_DIR, fname)
    data = torch.load(fpath, map_location="cpu")

    latents_list = []
    print(f"🟢 Processing file {file_idx}/{len(files)}: {fname} ({len(data)} items)\n")

    for item_idx, item in enumerate(data, 1):
        imgs = item["images"]
        item_latents = []
        print(f"  Item {item_idx}/{len(data)} → {len(imgs)} images")

        batch, batch_indices = [], []

        for img_idx, img_bytes in enumerate(imgs, 1):
            img_tensor = preprocess_image(img_bytes)
            batch.append(img_tensor)
            batch_indices.append(img_idx)

            # Encode when batch full or last image reached
            if len(batch) == BATCH_SIZE or img_idx == len(imgs):
                batch_tensors = torch.cat(batch, dim=0)
                torch.cuda.synchronize()
                t0 = time.time()
                z_batch = encode_batch(batch_tensors)
                torch.cuda.synchronize()
                dt = time.time() - t0

                for b_i, latent in zip(batch_indices, z_batch):
                    shape = tuple(latent.shape)
                    mem = latent.numel() * latent.element_size() / 1024**2
                    print(
                        f"    ✅ Encoded image {b_i}/{len(imgs)} "
                        f"→ latent {shape}, {latent.dtype}, ~{mem:.2f} MB "
                        f"({dt:.3f}s batch)"
                    )
                    item_latents.append(latent)

                batch.clear()
                batch_indices.clear()

        # Replace images with stacked latents
        if item_latents:
            item["latents"] = torch.stack(item_latents, dim=0)  # (N,C,H,W)
        else:
            item["latents"] = torch.empty(0)
        del item["images"]
        latents_list.append(item)
        gc.collect()

    # 🔹 Save combined latents as one .torch file
    out_path = os.path.join(OUTPUT_DIR, fname)
    torch.save(latents_list, out_path)

    del data, latents_list
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n💾 Finished {fname} ({file_idx}/{len(files)}) → saved {out_path}")
    print("-" * 70 + "\n")

print("✅ All files converted successfully and saved to:", OUTPUT_DIR)
