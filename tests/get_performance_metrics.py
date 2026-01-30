from types import SimpleNamespace
import sys
import os
project_root = os.path.abspath("..")
sys.path.append(project_root)

import torch
import yaml
import re
import glob
from pathlib import Path

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

from src.data.torch_latent_dataset import LVSMLatentDataset, EvalLVSMLatentDataset
from main import LVSMLauncher

from main import LVSMLauncher, LVSMLauncherConfig

from einops import rearrange
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

psnr_fn = PeakSignalNoiseRatio(data_range=1.0)
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
lpips_fn = LearnedPerceptualImagePatchSimilarity(
    net_type="alex", normalize=True
)


def get_config(overwrite):           
    
    dir = "../assets/vae_config.yaml" if overwrite.model_space == "VAE" else "../assets/px_config.yaml"
    raw = Path(dir).read_text()    
    clean = re.sub(r"!!python/[^ \n]+", "", raw)   # strip python tags

    config_dict = yaml.safe_load(clean)    
    cfg = LVSMLauncherConfig()

    for k, v in config_dict.items():
        if k != "model_config":   # we'll rebuild this separately
            setattr(cfg, k, v)
    
    m = config_dict["model_config"]

    # Enums were dumped as single-item lists
    ray_encoding = RayEncodingType(m["ray_encoding"][0])
    pos_enc      = PosEncType(m["pos_enc"][0])
    
    layer_cfg = m["encoder"]["layer"]

    encoder_layer = TransformerEncoderLayerConfig(
        d_model=layer_cfg["d_model"],
        nhead=layer_cfg["nhead"],
        dim_feedforward=overwrite.dim_feedforward,
        dropout=layer_cfg["dropout"],
        activation=torch.nn.functional.relu,  # original activation
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

    # Full model config
    model_cfg = LVSMDecoderOnlyModelConfig(
        ref_views=m["ref_views"],
        tar_views=m["tar_views"],
        encoder=encoder,
        img_shape=tuple([m["img_shape"][0]*overwrite.scale_factor, m["img_shape"][1]*overwrite.scale_factor,m["img_shape"][2]]),
        cam_shape=tuple([m["cam_shape"][0]*overwrite.scale_factor, m["cam_shape"][1]*overwrite.scale_factor,m["cam_shape"][2]]),        
        patch_size=overwrite.patch_size,        
        ray_encoding=ray_encoding,
        pos_enc=pos_enc,
    )
    
    cfg.model_config = model_cfg    
    return cfg

# ckpt_path = "/home/teampc/LVSM-VAE/results/nvs/VAE/release-1gpus-b8-s1-80k-CAMRAY-PROPE/ckpts/best_for_now/step-000870100.pt"
ckpt_path = "/home/teampc/LVSM-VAE/results/nvs/VAE/release-1gpus-b8-s1-80k-CAMRAY-PROPE/ckpts/best_for_now/step-000887200.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}

ow = SimpleNamespace()
ow.model_space = 'VAE'
ow.scale_factor = 1
ow.patch_size = 2
ow.supervise_views = 0
ow.dim_feedforward = 1024
ow.num_layers = 6

cfg = get_config(ow)
model = LVSMDecoderOnlyModel(cfg.model_config).to(device)
model.eval()

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

l = LVSMLauncher(LVSMLauncherConfig(model_space = "VAE"))
import time
import pandas as pd

data_path = "/mnt/hdd/data/re10k_full/test"
SUPERVISE_VIEWS = 3 # no effect

for INPUT_VIEWS in [2,4,8,16]:

    start = time.time()            

    eval_path = f"/home/teampc/LVSM-VAE/assets/evaluation_index_re10k_context{INPUT_VIEWS}.json"

    print(f"{time.time()-start} creating dataset")
    ds = EvalLVSMLatentDataset(torch_root=data_path, index_json_path=eval_path, input_views=INPUT_VIEWS, supervise_views=SUPERVISE_VIEWS)    
    print(f"{time.time()-start} creating dataloader")
    dataloader = torch.utils.data.DataLoader(ds,shuffle=False, num_workers=0)
    dataiter = iter(dataloader)

    psnr = 0
    ssim = 0
    lpips = 0

    n = 0
    # max_n = 5

    df = pd.DataFrame(columns=["ID", "PSNR", "SSIM", "LPIPS"])

    for data in tqdm.tqdm(dataloader, desc="Testing"):        
        
        n += 1
        # if n >= max_n: break

        data = next(dataiter)        
        processed = l.preprocess(data, input_views=INPUT_VIEWS)

        ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
        ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]    

        with torch.no_grad():
            out = model(ref_imgs.to(device), ref_cams, tar_cams)

        out_px = l.decode_tensors(out).squeeze(0).detach() 
        tar_px = l.decode_tensors(tar_imgs).squeeze(0).detach()          

        pred = rearrange(out_px.detach(),"B W H C -> B C W H").cpu()
        gt = rearrange(tar_px.detach(),"B W H C -> B C W H").cpu()

        s_id = data["image_path"][0][0]
        s_psnr = psnr_fn(pred,gt).item()
        s_ssim = ssim_fn(pred,gt).item()
        s_lpips = lpips_fn(pred,gt).item()
        

        psnr += s_psnr
        ssim += s_ssim
        lpips += s_lpips
        
        df.loc[len(df)] = [s_id, s_psnr, s_ssim, s_lpips]


    mean_psnr = psnr / n
    mean_ssim = ssim / n
    mean_lpips = lpips / n

    df.to_csv(f"/home/teampc/LVSM-VAE/notebooks/results/performance_metrics_4ctx/context{INPUT_VIEWS}.csv", index=False)
    with open(f"/home/teampc/LVSM-VAE/notebooks/results/performance_metrics_4ctx/context{INPUT_VIEWS}_res.txt", "w") as f:
        f.write(f"\npsn: {mean_psnr}\nssim: {mean_ssim}\nlpips: {mean_lpips}")


    print(f"metrics for {INPUT_VIEWS} input → 1 output:")
    print(f"\npsn: {mean_psnr}\nssim: {mean_ssim}\nlpips: {mean_lpips}")