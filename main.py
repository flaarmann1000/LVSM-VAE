import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from einops import rearrange
import wandb
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import logging

from src.data.dataset import TrainDataset, EvalDataset
from src.data.torch_dataset import LVSMDataset, EvalLVSMDataset
from src.data.latent_dataset import EvalLatentDataset, TrainLatentDataset
from src.data.torch_latent_dataset import LVSMLatentDataset, EvalLVSMLatentDataset

from src.models.lvsm_decoder_only import (
    Camera,
    LVSMDecoderOnlyModel,
    LVSMDecoderOnlyModelConfig,
)
from src.configs.lvsm_decoder_only_config import LVSMDecoderOnlyModelConfig

from src.utils.perceptual import Perceptual
from src.prope.utils.functional import random_SO3
from src.prope.utils.runner import Launcher, LauncherConfig, nested_to_device

from diffusers import AutoencoderKL



def write_tensor_to_disk(x: Tensor,path: str):
    torch.save(x.cpu(), path)    

def write_tensor_to_image(
    x: Tensor,
    path: str,
    downscale: int = 1,
    sqrt: bool = False,
    point: Tuple[int, int] = None,
):
    # x: [H, W, 1 or 3] in (0, 1)
    assert x.ndim == 3, x.shape
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 3)
    if sqrt:
        # reshape image to square
        h, w = x.shape[:2]
        if h > w:
            n_images = h // w
            n_sqrt = int(np.sqrt(n_images))
            x = rearrange(x, "(n1 n2 h) w c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
        elif h < w:
            n_images = w // h
            n_sqrt = int(np.sqrt(n_images))
            x = rearrange(x, "h (n1 n2 w) c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = (x * 255).to(torch.uint8).detach().cpu().numpy()
    if downscale > 1:
        image = cv2.resize(image, (0, 0), fx=1.0 / downscale, fy=1.0 / downscale)
    if point is not None:
        cv2.circle(image, point, 5, (255, 0, 0), -1)
    imageio.imsave(path, image)

@dataclass
class LVSMLauncherConfig(LauncherConfig):
    # model space config

    from_torch: int = 1
    grad_clip: float = 0.0
    data: str = ""
    
    const_lr: int = 1

    upscale: int = 1
    decode: int = 1
    overfit: int = 0
    norm: int = 0
    model_space: str = "PX" # can be VAE or PX

    # Dataset config
    dataset_patch_size: int = 256 # matters only for PX space
    dataset_supervise_views: int = 6
    dataset_input_views: int = 2
    dataset_batch_scenes: int = 4
    train_zoom_factor: float = 1.0 # matters only for PX space
    random_zoom: bool = False # matters only for PX space

    # Optimization config
    use_torch_compile: bool = True

    # Model config
    model_config: Any = field(
        default_factory=lambda: LVSMDecoderOnlyModelConfig(ref_views=2)
    )

    # Training config
    max_steps: int = 100000  # override
    ckpt_every: int = 1000  # override
    print_every: int = 100
    visual_every: int = 1000
    visual_wandb_every: int = 50000
    # lr: float = 4e-4
    lr: float = 5e-5
    warmup_steps: int = 2500

    # perceptual loss weight.
    perceptual_loss_w: float = 0.5

    # How many test scenes to run.
    test_every: int = 10000  # override
    test_n: Optional[int] = None
    test_input_views: int = 2
    test_supervise_views: int = 3
    test_zoom_factor: tuple[float, ...] = (1.0,)
    aug_with_world_origin_shift: bool = False
    aug_with_world_rotation: bool = False

    # Render a video
    render_video: bool = False

    # test index file
    test_index_fp: Optional[str] = None


class LVSMLauncher(Launcher):
    config: LVSMLauncherConfig    

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    vae32 = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="vae"    
    ).to(device)  


    def decode_tensors(self, t: Tensor) -> Tensor:
        """
        Accepts:
            [H, W, C]
            [V, H, W, C]
            [B, V, H, W, C]

        Returns:
            same leading shape but decoded to RGB.
        """
        t = t.to(self.device).float()        
        orig_shape = t.shape[:-3]   # could be (), (V), or (B,V)
        H, W, C = t.shape[-3:]        
        t = t.reshape(-1, H, W, C)      # [N, H, W, C] # flatten leading dims        
        t = t.permute(0, 3, 1, 2)       # [N, C, H, W]
        with torch.no_grad():
            out = self.vae32.decode(t).sample  # [N, 3, H_up, W_up]
        out = (out + 1) / 2
        out = out.clamp(0, 1)
        out = out.permute(0, 2, 3, 1)          # [N, H_up, W_up, 3]        
        out = out.reshape(*orig_shape, out.shape[1], out.shape[2], 3) # restore original leading dims
        return out


    # Data preprocessing.
    def preprocess(
        self, data: Dict, input_views: int
    ) -> Tuple[Tensor, Camera, Camera, Tensor]:
        data = nested_to_device(data, self.device)

        if (self.config.model_space == "PX"):
            images = data["image"] / 255.0
        else:
            images = (data["image"])
        
        Ks = data["K"]
        camtoworlds = data["camtoworld"]
        image_paths = data["image_path"]
        assert images.ndim == 5, images.shape
        n_batch, n_views, height, width, _ = images.shape

        # random shift and rotate.
        aug = torch.eye(4, device=self.device).repeat(n_batch, 1, 1)
        if self.config.aug_with_world_origin_shift:
            shifts = torch.randn((n_batch, 3), device=self.device)
            aug[:, :3, 3] = shifts
        if self.config.aug_with_world_rotation:
            rotations = random_SO3((n_batch,), device=self.device)
            aug[:, :3, :3] = rotations
        camtoworlds = torch.einsum("bij,bvjk->bvik", aug, camtoworlds)

        ref_imgs = images[:, :input_views]
        tar_imgs = images[:, input_views:]
        ref_cams = Camera(
            K=Ks[:, :input_views],
            camtoworld=camtoworlds[:, :input_views],
            width=width,
            height=height,
        )
        tar_cams = Camera(
            K=Ks[:, input_views:],
            camtoworld=camtoworlds[:, input_views:],
            width=width,
            height=height,
        )
        ref_paths = np.array(image_paths)[:input_views]
        tar_paths = np.array(image_paths)[input_views:]

        processed = {
            "ref_imgs": ref_imgs,
            "tar_imgs": tar_imgs,
            "ref_cams": ref_cams,
            "tar_cams": tar_cams,
            "ref_paths": ref_paths,
            "tar_paths": tar_paths,
        }
        return processed

    def train_initialize(self) -> Dict[str, Any]:
        # ------------- Setup Data. ------------- #
        
        root = f"train-overfit-{self.config.overfit}" if (self.config.overfit) else "train"
        
        if(self.config.data == ""):            
            if (self.config.model_space == "PX"):
                folder = f"./data/re10k_subset/{root}" if self.config.from_torch else f"./data/data_processed/realestate10k/{root}"
            else:
                folder = f"./data/re10k_subset_latent/{root}" if self.config.from_torch else f"./data/data_processed/realestate10k_latent/{root}/"
        else:            
            folder = f"{self.config.data}/{root}"
        
        scenes = sorted(glob.glob(f"{folder}/*"))
        print(f"Root folder: ./data/re10k_subset/{root}, approx number of train scenes: {len(scenes)} (chunks x 100)")
        
        if (self.config.model_space == "PX"):                        
            if self.config.from_torch:
                dataset = LVSMDataset(
                    folder,
                    square_crop=True,
                    patch_size=self.config.dataset_patch_size,
                    num_views=self.config.dataset_supervise_views + self.config.dataset_input_views,
                )
            else:
                dataset = TrainDataset(
                    scenes,
                    patch_size=self.config.dataset_patch_size,
                    zoom_factor=self.config.train_zoom_factor,
                    random_zoom=self.config.random_zoom,
                    supervise_views=self.config.dataset_supervise_views,
                )
        else:         
            if self.config.from_torch:
                dataset = LVSMLatentDataset(
                    folder,
                    input_views=cfg.dataset_input_views,
                    supervise_views=cfg.dataset_supervise_views,
                )
            else:
                dataset = TrainLatentDataset(
                    scenes,
                    supervise_views=self.config.dataset_supervise_views,
                    upscale_factor=self.config.upscale
                )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.dataset_batch_scenes,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.logging_on_master(f"Total scenes: {len(dataset)}")

        wandb.init(
            project="adlcv",
            config=self.config.__dict__,
            name=f"lvsm-decoder-only-{self.config.model_config.ray_encoding}-{self.config.model_config.pos_enc}-{self.config.max_steps} steps",
            tags=[
                self.config.model_space, 
                f"O{self.config.overfit}", 
                f"patch_size {self.config.model_config.patch_size}", 
                self.config.model_config.ray_encoding, 
                self.config.model_config.pos_enc, 
                "from_torch" if self.config.from_torch else "single_files",
                "const_lr" if self.config.const_lr else "fancy_lr"
                ]
        )

        # ------------- Setup Model. ------------- #
        model = LVSMDecoderOnlyModel(self.config.model_config).to(self.device)

        wandb.watch(model, log="gradients", log_freq=500)
        # Apply torch.compile for performance optimization if enabled
        if self.config.use_torch_compile:
            model = torch.compile(model)
        if (self.config.perceptual_loss_w > 0) and (self.config.model_space == "PX"):
            perceptual = Perceptual().to(self.device)
        else:
            perceptual = None
        print(f"Model is initialized in rank {self.world_rank}")

        # ------------- Setup Optimizer. ------------- #
        # Paper A.1 "We use a weight decay of 0.05 on all parameters except
        # the weights of LayerNorm layers."
        params_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" not in n],
            "weight_decay": 0.05,
        }
        params_no_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" in n],
            "weight_decay": 0.0,
        }
        optimizer = torch.optim.AdamW(
            [params_decay, params_no_decay], lr=self.config.lr, betas=(0.9, 0.95)
        )

        if self.config.const_lr:
             scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: 1.0  # keep LR constant
            )
        else:
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.01,
                        total_iters=self.config.warmup_steps,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=self.config.max_steps - self.config.warmup_steps,
                    ),
                ]
            )

        # ------------- Setup Metrics. ------------- #
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # Note: careful when comparing with papers: "vgg" or "alex"
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        # prepare returns
        state = {
            "model": model,
            "perceptual": perceptual,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "dataloader": dataloader,
            "dataiter": iter(dataloader),
            "ssim_fn": ssim_fn,
            "psnr_fn": psnr_fn,
            "lpips_fn": lpips_fn,
        }
        print(f"Launcher(train) is intialized in rank {self.world_rank}")

        self.start_time = time.time()
        self.last_time = self.start_time
        self.test_start = self.start_time


        return state

    def train_iteration(
        self, step: int, state: Dict[str, Any], acc_step: int, *args, **kwargs
    ) -> None:
        dataloader = state["dataloader"]
        dataiter = state["dataiter"]
        perceptual = state["perceptual"]
        model = state["model"]
        model.train()

        try:
            data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            data = next(dataiter)
            state["dataiter"] = dataiter

        input_views = data["K"].shape[1] - self.config.dataset_supervise_views        
        processed = self.preprocess(data, input_views=input_views)
        ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
        ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]        

        # Forward.
        with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
            outputs = model(ref_imgs, ref_cams, tar_cams)
            if (self.config.model_space == "PX"):
                outputs = torch.sigmoid(outputs)
            mse = F.mse_loss(outputs, tar_imgs)
            
            # REG
            if self.config.model_space == "VAE":
                mse += 1e-4 * outputs.square().mean()

            if (self.config.perceptual_loss_w > 0) and (self.config.model_space == "PX"):
                perceptual_loss = perceptual(
                    rearrange(outputs, "b v h w c -> (b v) c h w"),
                    rearrange(tar_imgs, "b v h w c -> (b v) c h w"),
                )
                loss = mse + perceptual_loss * self.config.perceptual_loss_w
            else:
                loss = mse

        # Loggings.
        if (
            self.config.visual_every > 0
            and step % self.config.visual_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            if self.config.model_space == "PX":
                write_tensor_to_image(
                    rearrange(outputs, "b v h w c-> (b h) (v w) c"),
                    f"{self.visual_dir}/outputs{step}.png",
                )
                write_tensor_to_image(
                    rearrange(tar_imgs, "b v h w c-> (b h) (v w) c"),
                    f"{self.visual_dir}/gts{step}.png",
                )
                write_tensor_to_image(
                    rearrange(ref_imgs, "b v h w c-> (b h) (v w) c"),
                    f"{self.visual_dir}/inputs{step}.png",
                )
            else:
                write_tensor_to_disk(outputs, f"{self.visual_dir}/outputs{step}.pt")                
                write_tensor_to_disk(tar_imgs, f"{self.visual_dir}/gt{step}.pt")
                write_tensor_to_disk(ref_imgs, f"{self.visual_dir}/inputs{step}.pt")

        if (
            self.config.visual_every > 0
            and step % self.config.visual_wandb_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            if self.config.model_space == "PX":
                wandb.log({f"test/output_after_{step}_steps": wandb.Image(outputs[0].detach().cpu().numpy())}, step=step)
            else:                                         
                img = self.decode_tensors(outputs[0].detach())
                wandb.log({f"test/output_after_{step}_steps": wandb.Image(img.cpu().numpy())}, step=step)
                # target = self.decode_tensors(tar_imgs[0].detach())                
                # wandb.log({f"test/target_after_{step}_steps": wandb.Image(target.cpu().numpy())}, step=step)                
        if (
            step % self.config.print_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            mse = F.mse_loss(outputs, tar_imgs)
            outputs = rearrange(outputs, "b v h w c-> (b v) c h w")
            tar_imgs = rearrange(tar_imgs, "b v h w c-> (b v) c h w")
            psnr = state["psnr_fn"](outputs, tar_imgs)
            ssim = state["ssim_fn"](outputs, tar_imgs)
            if (self.config.model_space == "PX"):
                lpips = state["lpips_fn"](outputs, tar_imgs)
            else:
                lpips = 0
            self.logging_on_master(
                f"Step: {step}, Loss: {loss:.3f}, PSNR: {psnr:.3f}, "
                f"SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}, "
                f"LR: {state['scheduler'].get_last_lr()[0]:.3e}"
            )
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/psnr", psnr, step)
            self.writer.add_scalar("train/ssim", ssim, step)
            self.writer.add_scalar("train/lpips", lpips, step)

            now = time.time()
            cur_total_time = now - self.start_time
            step_total_time = now - self.last_time
            self.last_time = now


            # Additional wandb logging
            wandb.log({
                "train/loss": loss.item(),
                "train/psnr": psnr.item(),
                "train/ssim": ssim.item(),
                "train/lpips": lpips.item() if self.config.model_space == "PX" else 0,
                "train/lr": state["scheduler"].get_last_lr()[0],
                "train/cum_time_in_seconds": cur_total_time,
                "train/step_time_in_seconds": step_total_time,
            }, step=step)
            
        return loss

    def test_initialize(
        self,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        # ------------- Setup Data. ------------- #
        dataset = None
        dataloaders = dict()
        if not self.config.render_video and self.config.test_index_fp is None:
            assert (
                self.config.test_input_views == 2
                and self.config.test_supervise_views == 3
            ), "Invalid input views and supervise views for RE10K, should be 2 and 3 respectively."
        
        root = f"test-overfit-{self.config.overfit}" if (self.config.overfit) else "test"
        
        if(self.config.data == ""):            
            if (self.config.model_space == "PX"):
                folder = f"./data/re10k_subset/{root}" if self.config.from_torch else f"./data/data_processed/realestate10k/{root}"
            else:
                folder = f"./data/re10k_subset_latent/{root}" if self.config.from_torch else f"./data/data_processed/realestate10k_latent/{root}/"
        else:            
            folder = f"{self.config.data}/{root}"                    
        
        for zoom_factor in self.config.test_zoom_factor:
            if (self.config.model_space == "PX"):
                if self.config.from_torch:
                    dataset = EvalLVSMDataset(
                        torch_root=folder,
                        square_crop=True,
                        patch_size=self.config.dataset_patch_size,
                        index_json_path = self.config.test_index_fp
                    )
                else:
                    dataset = EvalDataset(
                        folder=folder,
                        patch_size=self.config.dataset_patch_size,
                        zoom_factor=zoom_factor,
                        first_n=self.config.test_n,
                        rank=self.world_rank,
                        world_size=self.world_size,
                        input_views=self.config.test_input_views,
                        supervise_views=self.config.test_supervise_views,
                        render_video=self.config.render_video,
                        test_index_fp=self.config.test_index_fp,                                        
                    )
            else:
                if self.config.from_torch:
                    dataset = EvalLVSMLatentDataset(
                        torch_root=folder,
                        index_json_path=cfg.test_index_fp,
                    )
                else:
                    dataset = EvalLatentDataset(
                        folder=folder,                                 
                        first_n=self.config.test_n,
                        rank=self.world_rank,
                        world_size=self.world_size,
                        input_views=self.config.test_input_views,
                        supervise_views=self.config.test_supervise_views,
                        render_video=self.config.render_video,
                        test_index_fp=self.config.test_index_fp,    
                        upscale_factor=self.config.upscale                
                    )
            dataloaders[f"zoom{zoom_factor}"] = (
                self.config.test_input_views,
                torch.utils.data.DataLoader(
                    dataset, batch_size=1, num_workers=2, pin_memory=True
                ),
            )
        self.logging_on_master(f"Total scenes: {len(dataset)}")

        # ------------- Setup Model. ------------- #
        if model is None:
            model = LVSMDecoderOnlyModel(self.config.model_config).to(self.device)
            # Apply torch.compile for performance optimization if enabled
            if self.config.use_torch_compile:
                model = torch.compile(model)
            print(f"Model is initialized in rank {self.world_rank}")

        # ------------- Setup Metrics. ------------- #
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # Note: careful when comparing with papers: "vgg" or "alex"
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        # prepare returns
        state = {
            "model": model,
            "dataloaders": dataloaders,
            "psnr_fn": psnr_fn,
            "ssim_fn": ssim_fn,
            "lpips_fn": lpips_fn,
        }
        print(f"Launcher(Test) is intialized in rank {self.world_rank}")
        return state

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Dict[str, Any]) -> None:        
        dataloaders = state["dataloaders"]
        model = state["model"]
        model.eval()
        
        for label, (input_views, dataloader) in dataloaders.items():            
            psnrs, lpips, ssims, mses = [], [], [], []
            canvas = []  # for visualization
            for data in tqdm.tqdm(dataloader, desc="Testing"):                
                processed = self.preprocess(data, input_views=input_views)                
                ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
                ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]
                ref_paths, tar_paths = processed["ref_paths"], processed["tar_paths"]                
                # Forward.
                self.test_start = time.time()
                with torch.amp.autocast(
                    "cuda", enabled=self.config.amp, dtype=self.amp_dtype
                ):
                    outputs = model(ref_imgs, ref_cams, tar_cams)                
                if (self.config.model_space == "PX"):
                    outputs = torch.sigmoid(outputs)

                mse = F.mse_loss(outputs, tar_imgs)
                mses.append(mse.item())
                
                                
                inference_time =  time.time() - self.test_start                 

                if self.config.model_space == "VAE" and self.config.decode:
                    ref_imgs = self.decode_tensors(ref_imgs.detach())
                    outputs = self.decode_tensors(outputs.detach())
                    tar_imgs = self.decode_tensors(tar_imgs.detach())

                if self.config.render_video:
                    assert outputs.shape[0] == 1
                    path_splits = tar_paths[0, 0].split("/")
                    scene_name = path_splits[-3]
                    # dump video using imageio
                    imageio.mimwrite(
                        f"{self.test_dir}/{scene_name}.mp4",
                        (outputs[0].cpu().numpy() * 255).astype(np.uint8),
                        format="ffmpeg",
                        fps=15,
                    )
                else:
                    # dump images.
                    if len(canvas) < 10:
                        canvas_left = rearrange(ref_imgs, "b v h w c -> (b h) (v w) c")
                        canvas_right = rearrange(
                            torch.cat([tar_imgs, outputs], dim=3),
                            "b v h w c -> (b h) (v w) c",
                        )
                        channels = 3 if self.config.model_space == "PX" or self.config.decode else 16                        
                        canvas_mid = torch.ones(
                            len(canvas_left), 20, channels, device=self.device
                        )
                        canvas.append(
                            torch.cat([canvas_left, canvas_mid, canvas_right], dim=1)
                        )                    
                    # metrics.
                    outputs = rearrange(outputs, "b v h w c -> (b v) c h w")
                    tar_imgs = rearrange(tar_imgs, "b v h w c -> (b v) c h w")
                    psnrs.append(state["psnr_fn"](outputs, tar_imgs))
                    ssims.append(state["ssim_fn"](outputs, tar_imgs))
                    if (self.config.model_space == "PX") or self.config.decode:
                        if torch.isnan(outputs).any():  
                            lpips.append(0.0)
                        else: 
                            lpips.append(state["lpips_fn"](outputs, tar_imgs))
                    else:
                        lpips.append(0.0)                        
            
            if self.config.render_video:
                return

            # dump canvas.
            canvas = torch.cat(canvas, dim=0)
            if self.config.model_space == "PX" or self.config.decode:
                write_tensor_to_image(
                    canvas, f"{self.test_dir}/rank{self.world_rank}_{label}views.png"
                )        

            def distributed_avg(data: List[float], name: str) -> float:
                # collect metric from all ranks
                tensor_data = torch.tensor(data, device=self.device)

                if self.world_size > 1 and torch.distributed.is_initialized():
                    # Collect metric from all ranks
                    collected_sizes = [None] * self.world_size
                    torch.distributed.all_gather_object(collected_sizes, len(data))

                    collected = [torch.empty(size, device=self.device) for size in collected_sizes]
                    torch.distributed.all_gather(collected, tensor_data)
                    collected = torch.cat(collected)
                else:
                    # Single-GPU
                    collected = tensor_data
                self.logging_on_master(
                        f"Inf found in {label} views, {sum(torch.isinf(collected))} inf values for {name}."
                    )
                
                if torch.isnan(collected).any():
                    self.logging_on_master(
                        f"NaN found in {label} views, {sum(torch.isnan(collected))} nan values for {name}."
                    )
                    collected = collected[~torch.isnan(collected)]

                avg = collected.mean().item()
                return avg, len(collected)

            avg_psnr, n_total = distributed_avg(psnrs, "psnr")
            if self.config.model_space == "PX" or self.config.decode:
                avg_lpips, n_total = distributed_avg(lpips, "lpips")
            else:
                avg_lpips, n_total = (0,0)
            avg_ssim, n_total = distributed_avg(ssims, "ssim")
            avg_mse, n_total = distributed_avg(mses, "mses")

            self.logging_on_master(
                f"PSNR{label}: {avg_psnr:.3f}, SSIM{label}: {avg_ssim:.3f}, LPIPS{label}: {avg_lpips:.3f} "
                f"evaluated on {n_total} scenes at step {step}."
            )

            if self.world_rank == 0:
                self.writer.add_scalar(f"test/psnr{label}", avg_psnr, step)
                self.writer.add_scalar(f"test/ssim{label}", avg_ssim, step)
                self.writer.add_scalar(f"test/lpips{label}", avg_lpips, step)
                self.writer.add_scalar(f"test/mse{label}", avg_mse, step)
                self.writer.add_scalar(f"test/inference_time{label}", inference_time, step)
                with open(f"{self.test_dir}/metrics.json", "w") as f:
                    json.dump(
                        {
                            "label": label,
                            "step": step,
                            "n_total": n_total,
                            "psnr_px": avg_psnr,
                            "ssim_px": avg_ssim,
                            "lpips_px": avg_lpips,
                            "mse": avg_mse,
                            "inference_time": inference_time,
                        },
                        f,
                    )

                wandb.log({
                    f"test/{label}_psnr_px": avg_psnr,
                    f"test/{label}_ssim_px": avg_ssim,
                    f"test/{label}_lpips_px": avg_lpips,
                    f"test/{label}_mse": avg_mse,
                    f"test/{label}_inference_time_in_seconds": inference_time,
                    "test/step": step,
                    }, step=step)
                
                # Convert canvas tensor to numpy for wandb
                if self.config.model_space == "PX" or self.config.decode:
                    canvas_np = (canvas.detach().cpu().numpy() * 255).astype(np.uint8)
                    wandb.log({
                    f"test/{step}_image": wandb.Image(canvas_np)
                    }, step=step)

if __name__ == "__main__":
    """Example usage:

    # 2GPUs dry run
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        main.py lvsm-dry-run --model_config.encoder.num_layers 2
    """

    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")

    configs = {
        "lvsm": (
            "feedforward large view synthesis model",
            LVSMLauncherConfig(),
        ),        
        "lvsm-dry-run": (
            "dry run",
            LVSMLauncherConfig(
                model_space="PX",
                amp=True,
                amp_dtype="fp16",
                dataset_batch_scenes=1,
                max_steps=10,
                test_every=5,
                test_n=10,
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    
    if cfg.model_space == "VAE":
        # Override the defaults in cfg.model_config
        s = cfg.upscale
        cfg.model_config.img_shape = (32*s, 32*s, 16)
        cfg.model_config.cam_shape = (32*s, 32*s, 6)
        #  now defined in nvs.sh
        # cfg.model_config.patch_size = 1
        # cfg.model_config.patch_size = 2
        # cfg.model_config.patch_size = 8

    if cfg.norm: cfg.model_config.norm = cfg.norm

    prefix = "t" if cfg.from_torch else ""

    if cfg.overfit:
        cfg.test_n = None
        cfg.test_index_fp= f"overfitting_index_re10k-{prefix}{cfg.overfit}.json"
    else:
        if cfg.data = "":
            cfg.test_index_fp= f"evaluation_index_re10k_subset_{prefix}.json"
        else:
            cfg.test_index_fp= f"{cfg.data}/evaluation_index_re10k.json"
    
    launcher = LVSMLauncher(cfg)
    launcher.run()