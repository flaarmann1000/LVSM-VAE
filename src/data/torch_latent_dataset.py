import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset


# ============================================================
# Pose normalization (identical to your old latent dataset)
# ============================================================

def _normalize_poses_identity_unit_distance(
    in_c2ws: torch.Tensor,
    ref0_idx: int,
    ref1_idx: int,
):
    ref0_c2w = in_c2ws[ref0_idx]
    c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(ref0_c2w), in_c2ws)

    ref0_c2w = c2ws[ref0_idx]
    ref1_c2w = c2ws[ref1_idx]
    dist = torch.linalg.norm(ref1_c2w[:3, 3] - ref0_c2w[:3, 3])

    if dist > 1e-6:
        c2ws[:, :3, 3] /= dist

    return c2ws


# ============================================================
# Latent tensor normalization helper
# ============================================================

def _as_chw_latent(z: torch.Tensor) -> torch.Tensor:
    """
    Normalize latent to [C,H,W].

    Accepts:
      - [C,H,W]
      - [1,C,H,W]   (most common, saved from VAE)
    """
    if not torch.is_tensor(z):
        z = torch.as_tensor(z)

    if z.ndim == 4 and z.shape[0] == 1:
        z = z.squeeze(0)

    if z.ndim != 3:
        raise RuntimeError(f"Invalid latent shape {tuple(z.shape)}, expected [C,H,W]")

    return z


# ============================================================
# TRAIN DATASET (Iterable)
# ============================================================

class LVSMLatentDataset(IterableDataset):
    def __init__(
        self,
        roots,
        input_views: int = 2,
        supervise_views: int = 6,
        min_frame_dist: int = 1,
        max_frame_dist: Optional[int] = None,
    ):
        super().__init__()
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.num_views = input_views + supervise_views
        self.min_frame_dist = min_frame_dist
        self.max_frame_dist = max_frame_dist

        if isinstance(roots, (str, Path)):
            roots = [roots]

        self.chunks = []
        for r in roots:
            self.chunks += sorted(Path(r).glob("*.torch"))

        assert self.chunks, "No latent .torch files found"


    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        chunks = self.chunks

        if worker is not None:
            chunks = chunks[worker.id :: worker.num_workers]

        for chunk_path in chunks:
            scenes = torch.load(chunk_path)
            random.shuffle(scenes)

            for scene in scenes:
                out = self._process_scene(scene)
                if out is not None:
                    yield out


    def _select_views(self, n: int) -> Optional[List[int]]:
        N = self.num_views

        # Need at least 2 refs and enough total frames to sample N views
        if n < N:
            return None  # skip short scenes entirely
        
        # if n < 2:
        #     return None

        # if n <= N:
        #     return list(range(n))

        max_d = self.max_frame_dist or (n - 1)
        max_d = min(max_d, n - 1)
        min_d = min(self.min_frame_dist, max_d)

        if max_d <= min_d:
            return sorted(random.sample(range(n), N))

        d = random.randint(min_d, max_d)
        s = random.randint(0, n - d - 1)
        e = s + d

        mids = list(range(s + 1, e))
        if len(mids) < self.supervise_views:
            return sorted(random.sample(range(n), N))

        sup = random.sample(mids, self.supervise_views)
        return [s, e] + sup


    def _process_scene(self, scene: Dict[str, Any]):
        images = scene["images"]
        cameras = scene["cameras"]

        n = len(images)
        idxs = self._select_views(n)
        if idxs is None:
            return None
        
        if len(idxs) != self.num_views:
            return None

        idxs = sorted(idxs)
        idxs = [idxs[0], idxs[-1]] + idxs[1:-1]

        # ---- latents ----
        latents = torch.stack([_as_chw_latent(images[i]) for i in idxs])  # [V,C,H,W]
        latents = latents.permute(0, 2, 3, 1).contiguous().clone()        # [V,H,W,C], owning

        # ---- poses ----
        c2ws, Ks = self._convert_poses(cameras[idxs])
        c2ws = _normalize_poses_identity_unit_distance(
            c2ws, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        
        Ks = Ks.contiguous().clone()
        c2ws = c2ws.contiguous().clone()

        return {
            "image": latents.float(),
            "K": Ks.float(),
            "camtoworld": c2ws.float(),
            "image_path": [scene["key"]] * len(idxs),
        }


    def _convert_poses(self, poses: torch.Tensor):
        b = poses.shape[0]

        intrinsics = torch.eye(3).repeat(b, 1, 1)
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        w2c = torch.eye(4).repeat(b, 1, 1)
        w2c[:, :3] = poses[:, 6:].view(b, 3, 4)

        return w2c.inverse(), intrinsics


    def __len__(self):
        return len(self.chunks) * 100  # approximate


# ============================================================
# EVAL DATASET (Indexed)
# ============================================================

class EvalLVSMLatentDataset(Dataset):
    def __init__(
        self,
        torch_root,
        index_json_path,
        input_views: int = 2,
        supervise_views: int = 3,
    ):
        self.input_views = input_views
        self.supervise_views = supervise_views

        with open(index_json_path, "r") as f:
            self.index = json.load(f)

        self.scenes = {}
        for p in Path(torch_root).glob("*.torch"):
            for s in torch.load(p):
                self.scenes[s["key"]] = s

        self.keys = sorted(k for k in self.index if k in self.scenes)


    def __len__(self):
        return len(self.keys)


    def __getitem__(self, idx: int):
        key = self.keys[idx]
        scene = self.scenes[key]
        spec = self.index[key]

        ids = spec["context"] + spec["target"]

        latents = torch.stack([_as_chw_latent(scene["images"][i]) for i in ids])
        latents = latents.permute(0, 2, 3, 1).contiguous().clone()

        c2ws, Ks = self._convert_poses(scene["cameras"][ids])
        c2ws = _normalize_poses_identity_unit_distance(
            c2ws, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        
        Ks = Ks.contiguous().clone()
        c2ws = c2ws.contiguous().clone()

        return {
            "image": latents.float(),
            "K": Ks.float(),
            "camtoworld": c2ws.float(),
            "image_path": [key] * len(ids),
            "scene_idx": idx,
        }


    def _convert_poses(self, poses: torch.Tensor):
        b = poses.shape[0]

        intrinsics = torch.eye(3).repeat(b, 1, 1)
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        w2c = torch.eye(4).repeat(b, 1, 1)
        w2c[:, :3] = poses[:, 6:].view(b, 3, 4)

        return w2c.inverse(), intrinsics
