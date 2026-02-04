from types import SimpleNamespace
import sys, os, re, yaml, json
from pathlib import Path

project_root = os.path.abspath("..")
sys.path.append(project_root)

import numpy as np
import torch
import imageio.v2 as imageio

from src.models.lvsm_decoder_only import LVSMDecoderOnlyModel
from src.configs.lvsm_decoder_only_config import (
    LVSMDecoderOnlyModelConfig,
    RayEncodingType,
    PosEncType,
)
from src.prope.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)
from src.data.torch_latent_dataset import EvalLVSMLatentDataset
from main import LVSMLauncher, LVSMLauncherConfig

# --------------------------
# Config loader
# --------------------------
def get_config(overwrite):
    cfg_path = "../assets/vae_config.yaml" if overwrite.model_space == "VAE" else "../assets/px_config.yaml"
    raw = Path(cfg_path).read_text()
    clean = re.sub(r"!!python/[^ \n]+", "", raw)  # strip python tags
    config_dict = yaml.safe_load(clean)

    cfg = LVSMLauncherConfig()
    for k, v in config_dict.items():
        if k != "model_config":
            setattr(cfg, k, v)

    m = config_dict["model_config"]
    ray_encoding = RayEncodingType(m["ray_encoding"][0])
    pos_enc = PosEncType(m["pos_enc"][0])

    layer_cfg = m["encoder"]["layer"]
    encoder_layer = TransformerEncoderLayerConfig(
        d_model=layer_cfg["d_model"],
        nhead=layer_cfg["nhead"],
        dim_feedforward=overwrite.dim_feedforward,
        dropout=layer_cfg["dropout"],
        activation=torch.nn.functional.relu,
        batch_first=layer_cfg["batch_first"],
        bias=layer_cfg["bias"],
        layer_norm_eps=layer_cfg["layer_norm_eps"],
        modulation_activation=layer_cfg["modulation_activation"],
        norm_first=layer_cfg["norm_first"],
        norm_type=layer_cfg["norm_type"],
        elementwise_affine=layer_cfg["elementwise_affine"],
        qk_norm=layer_cfg["qk_norm"],
    )
    encoder = TransformerEncoderConfig(
        layer=encoder_layer,
        num_layers=overwrite.num_layers,
        input_norm=m["encoder"]["input_norm"],
        output_norm=m["encoder"]["output_norm"],
        checkpointing=m["encoder"]["checkpointing"],
    )

    model_cfg = LVSMDecoderOnlyModelConfig(
        ref_views=m["ref_views"],
        tar_views=m["tar_views"],
        encoder=encoder,
        img_shape=(m["img_shape"][0] * overwrite.scale_factor, m["img_shape"][1] * overwrite.scale_factor, m["img_shape"][2]),
        cam_shape=(m["cam_shape"][0] * overwrite.scale_factor, m["cam_shape"][1] * overwrite.scale_factor, m["cam_shape"][2]),
        patch_size=overwrite.patch_size,
        ray_encoding=ray_encoding,
        pos_enc=pos_enc,
    )
    cfg.model_config = model_cfg
    return cfg


def save_rgb_png(path: str, x: torch.Tensor):
    """x: [H,W,3] float in [0,1]"""
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0 + 0.5).to(torch.uint8).numpy()
    imageio.imwrite(path, np.ascontiguousarray(x))


def write_mp4(path: str, frames_u8: np.ndarray, fps: int = 15):
    """
    frames_u8: [T,H,W,3] uint8 contiguous
    """
    frames_u8 = np.ascontiguousarray(frames_u8)
    imageio.mimwrite(
        path,
        frames_u8,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def main():
    # -------------
    # USER SETTINGS
    # -------------
    ckpt_path   = "/home/teampc/LVSM-VAE/results/nvs/VAE/release-1gpus-b8-s1-80k-CAMRAY-PROPE/ckpts/best_for_now/step-000903600.pt"
    # ckpt_path   = "/home/teampc/LVSM-VAE/results/nvs/VAE/release-1gpus-b8-s1-80k-CAMRAY-PROPE/ckpts/best_for_now/step-000870100.pt"
    data_root   = "/mnt/hdd/data/re10k_full/test"
    # index_json  = "/home/teampc/LVSM-VAE/assets/evaluation_index_re10k_video.json"
    # index_json  = "/home/teampc/LVSM-VAE/assets/evaluation_index_re10k_video_large_context.json"    
    index_json  = "/home/teampc/LVSM-VAE/assets/evaluation_index_re10k_video_large_context2.json"
    INPUT_VIEWS = 8
    SUPERVISE_VIEWS = 3
    INDEX = 0                     
    OUT_DIR = "./renders"
    FPS = 15
    BOOMERANG = True
    WRITE_DEBUG_PNGS = True

    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model cfg from yaml + overrides
    ow = SimpleNamespace()
    ow.model_space = "VAE"
    ow.scale_factor = 1
    ow.patch_size = 2
    ow.dim_feedforward = 1024
    ow.num_layers = 6

    cfg = get_config(ow)

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}

    model = LVSMDecoderOnlyModel(cfg.model_config).to(device).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Launcher only for preprocess+decode
    l = LVSMLauncher(LVSMLauncherConfig(model_space="VAE"))

    # Dataset
    ds = EvalLVSMLatentDataset(
        torch_root=data_root,
        index_json_path=index_json,
        input_views=INPUT_VIEWS,
        supervise_views=SUPERVISE_VIEWS,
    )

    assert 0 <= INDEX < len(ds), f"INDEX {INDEX} out of range (len={len(ds)})"
    data = ds[INDEX]

    
    def add_batch(d):
        out = {}
        for k, v in d.items():
            if torch.is_tensor(v):
                out[k] = v.unsqueeze(0)
            else:
                out[k] = [v] if isinstance(v, str) else v
        return out

    
    if torch.is_tensor(data["K"]) and data["K"].ndim == 3:  # [V,3,3] -> [1,V,3,3]
        data = add_batch(data)

    processed = l.preprocess(data, input_views=INPUT_VIEWS)
    ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
    ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]

    with torch.no_grad():
        out_lat = model(ref_imgs.to(device), ref_cams, tar_cams)

    # Decode to RGB
    out_px = l.decode_tensors(out_lat).squeeze(0)   # [T,H,W,3]
    tar_px = l.decode_tensors(tar_imgs).squeeze(0)  # [T,H,W,3]

    # Debug PNGs
    if WRITE_DEBUG_PNGS:
        save_rgb_png(os.path.join(OUT_DIR, f"idx{INDEX:05d}_ref0.png"), l.decode_tensors(ref_imgs).squeeze(0)[0])
        save_rgb_png(os.path.join(OUT_DIR, f"idx{INDEX:05d}_gt0.png"),  tar_px[0])
        save_rgb_png(os.path.join(OUT_DIR, f"idx{INDEX:05d}_out0.png"), out_px[0])

    # Build video frames (prediction)
    frames = out_px.detach().cpu().clamp(0, 1)
    frames_u8 = (frames * 255.0 + 0.5).to(torch.uint8).numpy()  # [T,H,W,3] uint8

    if BOOMERANG and frames_u8.shape[0] >= 3:
        frames_u8 = np.concatenate([frames_u8, frames_u8[-2:0:-1]], axis=0)

    # Name the video using scene id if available
    scene_id = None
    try:        
        scene_id = data["image_path"][0][0].split("/")[0]
    except Exception:
        scene_id = f"idx{INDEX:05d}"

    out_mp4 = os.path.join(OUT_DIR, f"{scene_id}_idx{INDEX:05d}_ctx{INPUT_VIEWS}.mp4")
    write_mp4(out_mp4, frames_u8, fps=FPS)
    print("Wrote:", out_mp4, "frames:", frames_u8.shape)


if __name__ == "__main__":
    main()
