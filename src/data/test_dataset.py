import json
import random
from io import BytesIO
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import IterableDataset



class LSVMDataset(IterableDataset):
    """
    Dataset that loads from .torch files (PixelSplat style) but outputs 
    in LVSM format with proper pose preprocessing.
    """
    
    def __init__(
        self,
        roots,
        num_views=4,
        image_size=256,
        patch_size=16,
        scene_scale_factor=1,
        square_crop=False,
        min_frame_dist=25,
        max_frame_dist=100,
        inference=False,
        view_idx_file_path=None,
        render=False
    ):
        super().__init__()
        self.to_tensor = tf.ToTensor()
        
        # Store parameters
        self.num_views = num_views
        self.image_size = image_size
        self.patch_size = patch_size
        self.scene_scale_factor = scene_scale_factor
        self.square_crop = square_crop
        self.min_frame_dist = min_frame_dist
        self.max_frame_dist = max_frame_dist
        self.inference = inference
        self.render = render
        # Collect .torch chunks from roots
        self.chunks = []
        if isinstance(roots, (str, Path)):
            roots = [roots]
        
        for root in roots:
            root = Path(root)
            if root.exists():
                chunks = sorted([p for p in root.iterdir() if p.suffix == ".torch"])
                self.chunks.extend(chunks)
        
        if not self.chunks:
            raise ValueError(f"No .torch files found in roots: {roots}")
        
        # Inference mode settings
        self.view_idx_list = {}
        if self.inference and view_idx_file_path:
            if Path(view_idx_file_path).exists():
                with open(view_idx_file_path, 'r') as f:
                    self.view_idx_list = json.load(f)
    
    def __iter__(self):
        # Shuffle chunks for training
        if not self.inference:
            indices = torch.randperm(len(self.chunks))
            chunks = [self.chunks[i] for i in indices]
        else:
            chunks = self.chunks
        
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            chunks = [c for i, c in enumerate(chunks) 
                     if i % worker_info.num_workers == worker_info.id]
        
        for chunk_path in chunks:
            # Load chunk
            chunk = torch.load(chunk_path)
            
            # Shuffle examples within chunk
            if not self.inference:
                indices = torch.randperm(len(chunk))
                chunk = [chunk[i] for i in indices]
            
            for example in chunk:
                try:
                    processed = self.process_example(example)
                    if processed is not None:
                        yield processed
                except Exception as e:
                    print(f"Error processing example {example.get('key', 'unknown')}: {e}")
                    continue
    
    def process_example(self, example):
        """Process a single example from the chunk."""
        scene_name = example["key"]
        
        # Get poses from cameras (format: [fx, fy, cx, cy, ..., w2c_matrix])
        cameras = example["cameras"]  # Shape: [num_views, 18]
        extrinsics, intrinsics = self.convert_poses(cameras)
        
        # Select view
        if self.inference and self.render:
            image_indices = torch.arange(len(cameras))

        elif self.inference and scene_name in self.view_idx_list:
            view_spec = self.view_idx_list[scene_name]
            if view_spec is None:
                return None
            image_indices = view_spec["context"] + view_spec["target"]
            image_indices = torch.tensor(image_indices)
        else:
            image_indices = self.view_selector(len(cameras))
            if image_indices is None:
                return None
        
        # Load images
        try:
            images_data = [example["images"][idx.item()] for idx in image_indices]
            images = self.convert_images(images_data)
        except (IndexError, KeyError):
            return None
        
        # Extract selected views
        selected_extrinsics = extrinsics[image_indices]
        selected_intrinsics = intrinsics[image_indices]
        
        # Preprocess images to match LVSM format
        images, intrinsics_processed = self.preprocess_frames(
            images, selected_intrinsics
        )
        
        # Convert extrinsics (w2c) to c2w for LVSM format
        c2ws = selected_extrinsics  # Already inverted in convert_poses
        
        # Preprocess poses (centerize and scale)
        c2ws = self.preprocess_poses(c2ws, self.scene_scale_factor)
        
        # Create indices tensor [view, 2] where each row is [image_idx, scene_idx]
        scene_idx = 0  # Will be set by DataLoader if needed
        image_indices_expanded = image_indices.long().unsqueeze(-1)
        scene_indices = torch.full_like(image_indices_expanded, scene_idx)
        indices = torch.cat([image_indices_expanded, scene_indices], dim=-1)
        
        return {
            "image": images,
            "camtoworld": c2ws,
            "K": intrinsics_processed,
            "index": indices,
            "scene_name": scene_name,
            "image_path": "./data/re10k_subset/" + scene_name # TODO: replace with correct path
        }
    
    def convert_poses(self, poses):
        """Convert from PixelSplat format to extrinsics and intrinsics."""
        b = poses.shape[0]
        
        # Extract intrinsics (normalized K matrix)
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        
        # Extract extrinsics (convert w2c to c2w)
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        c2w = w2c.inverse()
        
        return c2w, intrinsics
    
    def convert_images(self, images_data):
        """Convert byte tensors to torch images."""
        torch_images = []
        for img_data in images_data:
            img = Image.open(BytesIO(img_data.numpy().tobytes()))
            torch_images.append(self.to_tensor(img))
        return torch.stack(torch_images)
    
    def preprocess_frames(self, images, intrinsics):
        """Preprocess images and intrinsics to target size (LVSM style)."""
        processed_images = []
        processed_intrinsics = []
        
        for img, K in zip(images, intrinsics):
            # img shape: [3, H, W]
            _, original_h, original_w = img.shape
            
            # Calculate new width maintaining aspect ratio
            resize_w = int(self.image_size / original_h * original_w)
            resize_w = int(round(resize_w / self.patch_size) * self.patch_size)
            
            # Resize image
            img_resized = F.interpolate(
                img.unsqueeze(0), 
                size=(self.image_size, resize_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Update intrinsics
            resize_ratio_x = resize_w / original_w
            resize_ratio_y = self.image_size / original_h
            K_new = K.clone()
            K_new[0, 0] *= resize_ratio_x  # fx
            K_new[1, 1] *= resize_ratio_y  # fy
            K_new[0, 2] *= resize_ratio_x  # cx
            K_new[1, 2] *= resize_ratio_y  # cy
            
            # Square crop if needed
            if self.square_crop:
                min_size = min(self.image_size, resize_w)
                start_h = (self.image_size - min_size) // 2
                start_w = (resize_w - min_size) // 2
                img_resized = img_resized[:, start_h:start_h+min_size, start_w:start_w+min_size]
                img_resized = rearrange(img_resized, 'c h w -> h w c')
                K_new[0, 2] -= start_w
                K_new[1, 2] -= start_h
            
            processed_images.append(img_resized)
            # Convert K to fxfycxcy format for LVSM

            processed_intrinsics.append(K_new)
        
        images = torch.stack(processed_images, dim=0)
        intrinsics = torch.stack(processed_intrinsics, dim=0)
        return images, intrinsics
    
    def preprocess_poses(self, c2ws, scene_scale_factor=1.35):
        """Preprocess poses: align to mean camera and rescale scene."""
        # Center is the average of all camera centers
        center = c2ws[:, :3, 3].mean(0)
        
        # Average direction vectors
        avg_forward = F.normalize(c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)
        
        # Build average pose
        avg_pose = torch.eye(4, device=c2ws.device)
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose)  # Convert to w2c for alignment
        
        # Align all poses
        c2ws = avg_pose @ c2ws
        
        # Rescale scene
        scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale
        c2ws[:, :3, 3] /= scene_scale
        
        return c2ws
    
    def view_selector(self, num_frames):
        """
        Robust view selection.
        - Uses LVSM logic when possible
        - Falls back to simple uniform sampling for small scenes
        """

        # Absolute minimum: need at least 2 views
        if num_frames < 2:
            return None

        # If scene is too small for LVSM constraints → fallback
        if (
            num_frames < self.num_views
            or num_frames - 1 <= self.min_frame_dist
        ):
            # Uniformly sample available frames
            if num_frames >= self.num_views:
                indices = torch.linspace(
                    0, num_frames - 1, steps=self.num_views
                ).long()
            else:
                # Repeat frames if extremely small (overfit safety)
                indices = torch.arange(num_frames)
                indices = indices.repeat(
                    (self.num_views + num_frames - 1) // num_frames
                )[: self.num_views]

            return indices

        # ---- Original LVSM logic ----
        min_dist = self.min_frame_dist
        max_dist = min(num_frames - 1, self.max_frame_dist)

        frame_dist = random.randint(min_dist, max_dist)
        start_frame = random.randint(0, num_frames - frame_dist - 1)
        end_frame = start_frame + frame_dist

        middle_frames = random.sample(
            range(start_frame + 1, end_frame),
            self.num_views - 2
        )

        indices = [start_frame, end_frame] + middle_frames
        return torch.tensor(indices)

    
    def __len__(self):
        # Approximate length based on chunks
        return len(self.chunks) * 100  # Rough estimate