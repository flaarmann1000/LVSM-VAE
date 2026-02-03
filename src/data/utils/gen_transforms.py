# Code adapted from https://github.com/liruilong940607/prope

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO


def decode_jpeg_tensor(t):
    """
    t: 1D uint8 tensor containing JPEG bytes.
    Returns a PIL Image.
    """
    if not isinstance(t, torch.Tensor):
        raise ValueError("Image entry is not a tensor")

    arr = t.cpu().numpy().tobytes()  # convert raw bytes
    return Image.open(BytesIO(arr)).convert("RGB")


def process_torch_file(torch_path, output_root):
    """
    torch file contains:
    - list of dicts
      each dict contains {
         key: str,
         timestamps: Tensor[N],
         cameras: Tensor[N, 18],
         images: list of byte-tensors
      }
    """
    seq_list = torch.load(torch_path)

    for entry in tqdm(seq_list, desc=f"Processing {torch_path}"):
        seqname    = entry["key"]
        timestamps = entry["timestamps"].cpu().numpy()
        cameras    = entry["cameras"].cpu().numpy()
        images     = entry["images"]   # list of 1-D jpeg byte tensors

        seq_dir = os.path.join(output_root, seqname)
        img_dir = os.path.join(seq_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # -----------------------------------------
        # 1) Decode and save all images
        # -----------------------------------------
        decoded_images = []
        for ts, jpeg_bytes in zip(timestamps, images):
            img = decode_jpeg_tensor(jpeg_bytes)
            decoded_images.append(img)

            img.save(os.path.join(img_dir, f"{int(ts):06d}.png"))

        # -----------------------------------------
        # 2) Create transforms.json
        # -----------------------------------------
        H, W = decoded_images[0].size[1], decoded_images[0].size[0]

        transforms = {
            "h": H,
            "w": W,
            "fl_x":float(cameras[0, 0]),
            "fl_y":float(cameras[0, 1]),
            "cx": float(cameras[0, 2]),
            "cy": float(cameras[0, 3]),
            "frames": []
        }

        for ts, cam in zip(timestamps, cameras):
            pose = cam[6:].reshape(3, 4)

            # Convert world-to-camera → camera-to-world
            w2c = np.eye(4)
            w2c[:3, :4] = pose
            c2w = np.linalg.inv(w2c)

            transforms["frames"].append({
                "file_path": f"images/{int(ts):06d}.png",
                "transform_matrix": c2w.tolist()
            })

        with open(os.path.join(seq_dir, "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=4)

        print(f"✔ Finished: {seqname}")


if __name__ == "__main__":
    mode = "test"  # "train" or "test"
    input_folder = f"./data/re10k_subset/{mode}"    
    output_root  = f"./data/re10k_img/{mode}"

    os.makedirs(output_root, exist_ok=True)

    for name in os.listdir(input_folder):
        if name.endswith(".torch"):
            process_torch_file(os.path.join(input_folder, name), output_root)
