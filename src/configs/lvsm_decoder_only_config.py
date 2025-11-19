from dataclasses import dataclass, field
from typing import Literal, Tuple
from enum import Enum
import torch.nn.functional as F


from src.prope.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)

class RayEncodingType(str, Enum):
    PLUCKER = "plucker"
    CAMRAY = "camray"
    NONE = "none"
    RAYMAP = "raymap"

class PosEncType(str, Enum):
    PROPE = "prope"
    GTA = "gta"
    NONE = "none"


@dataclass
class LVSMDecoderOnlyModelConfig:

    ref_views: int
    tar_views: int = 1

    encoder: TransformerEncoderConfig = field(
        default_factory=lambda: TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=768,
                nhead=16,
                dim_feedforward=3072,
                dropout=0.0,
                activation=F.relu,
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True,
                bias=False,
                elementwise_affine=True,
                norm_type="layer_norm",
                modulation_activation=None,
                qk_norm=False,
            ),
            num_layers=6,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
    )

    img_shape: Tuple[int, ...] = (256, 256, 3)
    cam_shape: Tuple[int, ...] = (256, 256, 6)
    patch_size: int = 8

    # How the input rays are encoded.
    ray_encoding: RayEncodingType =  RayEncodingType.PLUCKER
    pos_enc: PosEncType = PosEncType.NONE