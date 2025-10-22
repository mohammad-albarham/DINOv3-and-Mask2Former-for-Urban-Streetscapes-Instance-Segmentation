"""
DINOv3-Mask2Former Small+ Model Implementation

This module provides a complete DINOv3-Mask2Former model with custom backbone replacement.
The model uses DINOv3 as the backbone and Mask2Former as the segmentation head.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from transformers import AutoModel, AutoModelForUniversalSegmentation, AutoConfig
import logging

logger = logging.getLogger(__name__)
from huggingface_hub import PyTorchModelHubMixin

from rich import print as rprint

import torch
import torch.nn as nn
from typing import Dict, List
from transformers import AutoModel, AutoModelForUniversalSegmentation, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.utils._typing import is_jsonable

# Assuming rich for rprint; fallback to print
try:
    from rich import print as rprint
except ImportError:
    rprint = print


class Adapter(nn.Module):
    """
    Adapter to project multi-layer DINOv3 features to Mask2Former-expected channels.
    Each layer gets its own projection (1x1 conv) for progressive channel matching.
    """
    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Assume features are already at common resolution; project each independently
        return [self.projections[i](feat) for i, feat in enumerate(features)]


class DinoV3WithAdapterBackbone(nn.Module):
    """
    Custom backbone: DINOv3 with SegDINO-inspired multi-layer extraction for semantic multi-scale.
    Extracts from layers [3,6,9,12], projects channels; all at native stride=16.
    """
    def __init__(self, model_name: str, out_channels: List[int]):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.adapter = Adapter(self.model.config.hidden_size, out_channels)

        # Output features named for Mask2Former
        self.out_features = [f"stage_{i}" for i in range(len(out_channels))]
        self._out_feature_channels = {
            name: ch for name, ch in zip(self.out_features, out_channels)
        }
        # Uniform stride for DINOv3 (ViT: all layers same res; patch_size=16)
        stride = 16  # Adjust to self.model.config.patch_size if needed
        self._out_feature_strides = {name: stride for name in self.out_features}

        # SegDINO-inspired: Extract from early/mid/late layers for semantic multi-scale
        self.layers_to_extract = [3, 6, 9, 12]  # Layers 0-indexed; +1 in hidden_states

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # DINOv3 forward with hidden states
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        # Spatial dims post-patch embed
        batch_size, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        feat_h, feat_w = height // patch_size, width // patch_size

        # Extract and reshape each selected layer (ignore CLS token)
        extracted_features = []
        for layer_idx in self.layers_to_extract:
            layer_output = hidden_states[layer_idx + 1]  # hidden_states[0] is embed, [1]=layer0
            feature_map = (
                layer_output[:, 1:, :]  # Remove CLS
                .permute(0, 2, 1)
                .reshape(batch_size, self.model.config.hidden_size, feat_h, feat_w)
            )
            # SegDINO: Align to common res (all already same, but bilinear if needed for variants)
            # If future variants have varying res, add: F.interpolate(feature_map, size=(feat_h, feat_w), mode='bilinear')
            extracted_features.append(feature_map)

        # Project channels per layer (progressive: early layer -> fewer channels)
        adapted_features = self.adapter(extracted_features)

        # Return dict for Mask2Former
        return {name: feat for name, feat in zip(self.out_features, adapted_features)}


class Mask2Former_Dinov3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        dinov3_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",  # Base/16 for hidden=768; adjust as needed
        expected_channels: List[int] = [96, 192, 384, 768],
        freeze_backbone: bool = True,
        hub_token: str = None,
    ):
        super().__init__()

        print(f"is_jsonable label2id: {is_jsonable(label2id)}")

        self.label2id = label2id
        self.id2label = id2label
        self.dinov3_model_name = dinov3_model_name
        self.expected_channels = expected_channels
        self.freeze_backbone = freeze_backbone
        self.hub_token = hub_token

        mask2former_model_name = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        model = AutoModelForUniversalSegmentation.from_pretrained(
            mask2former_model_name,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            token=hub_token,
        )
        custom_backbone = DinoV3WithAdapterBackbone(dinov3_model_name, expected_channels)
        model.model.backbone = custom_backbone
        if freeze_backbone:
            for param in model.model.backbone.model.parameters():
                param.requires_grad = False

        # Config updates for uniform DINOv3 strides (SegDINO-style semantic multi-scale)
        model.config.feature_strides = [16, 16, 16, 16]
        model.config.common_stride = 16
        # Optional: Adjust pixel decoder if needed (e.g., in_maskformer_self_attention for fusion)

        # HF compatibility configs
        dino_config = AutoConfig.from_pretrained(dinov3_model_name)
        dino_config.architectures = ["DinoV3WithAdapterBackbone"]
        mask2former_config = AutoConfig.from_pretrained(mask2former_model_name)
        mask2former_config.model_type = "mask2former-with-dinov3-segdino"
        mask2former_config.backbone_config = dino_config
        model.config = mask2former_config

        self.inner_model = model
        self.config = model.config

        # Param count
        trainable_params = sum(p.numel() for p in self.inner_model.parameters() if p.requires_grad)
        rprint(f"Number of trainable parameters after optimizations: {trainable_params}")

        # Debug breakdown
        total_params = sum(p.numel() for p in self.inner_model.parameters())
        backbone_total = sum(p.numel() for p in model.model.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in model.model.backbone.parameters() if p.requires_grad)
        head_trainable = trainable_params - backbone_trainable
        rprint(f"Total params: {total_params}")
        rprint(f"Backbone total: {backbone_total}, trainable: {backbone_trainable} (adapter mainly)")
        rprint(f"Head trainable: {head_trainable}")

        # Optional: List top trainable modules
        # trainable_modules = {name: sum(p.numel() for p in module.parameters() if p.requires_grad) 
        #                     for name, module in model.named_modules() if sum(p.numel() for p in module.parameters() if p.requires_grad) > 100000}
        # for name, count in trainable_modules.items():
        #     rprint(f"{name}: {count}")

        # Optional details
        # for name, param in self.inner_model.named_parameters():
        #     if param.requires_grad:
        #         rprint(f"{name}: {param.shape}")

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


def create_mask2former_dinov3_model(
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    dinov3_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    expected_channels: List[int] = [96, 192, 384, 768],
    freeze_backbone: bool = True,
    hub_token: str = None,
) -> AutoModelForUniversalSegmentation:
    """
    Create a complete DINOv3-Mask2Former model with custom backbone replacement.

    Args:
        label2id: Dictionary mapping label names to IDs
        id2label: Dictionary mapping IDs to label names
        dinov3_model_name: HuggingFace model name for DINOv3
        expected_channels: List of output channels for each stage
        freeze_backbone: Whether to freeze DINOv3 backbone weights
        hub_token: HuggingFace Hub token if needed

    Returns:
        Complete DINOv3-Mask2Former model ready for training/inference
    """
    # Fixed Mask2Former base model
    mask2former_model_name = "facebook/mask2former-swin-small-coco-instance"

    logger.info(f"Creating DINOv3-Mask2Former model...")
    logger.info(f"  - Mask2Former base: {mask2former_model_name}")
    logger.info(f"  - DINOv3 backbone: {dinov3_model_name}")
    logger.info(f"  - Expected channels: {expected_channels}")
    logger.info(f"  - Freeze backbone: {freeze_backbone}")

    # TODO: The model here loaded into the cpu, check it!
    # TODO: Why we use AutoModelForUniversalSegmentation instead of AutoModelForInstanceSegmentation
    # 1. Load the base Mask2Former model
    model = AutoModelForUniversalSegmentation.from_pretrained(
        mask2former_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=hub_token,
    )

    # 2. Create custom DINOv3 backbone with adapter
    custom_backbone = DinoV3WithAdapterBackbone(dinov3_model_name, expected_channels)

    # 3. Replace the backbone
    model.model.backbone = custom_backbone

    # 4. Freeze DINOv3 weights if requested
    if freeze_backbone:
        for param in model.model.backbone.model.parameters():
            param.requires_grad = False
        logger.info("DINOv3 backbone weights frozen.")
    else:
        logger.info("DINOv3 backbone weights remain trainable.")

    logger.info("Successfully created DINOv3-Mask2Former model.")

    # 5. update the model configuration as well:

    # model.config.backbone = "dinov3"
    model.config.model_type = "mask2former-with-dinov3"
    # Optionally, add more info about your adapter design, layer indices, etc.

    # Download the config automatically from Hugging Face
    dino_config = AutoConfig.from_pretrained(dinov3_model_name)

    # Now you can inspect, modify, and save it:
    print(f"print Dino model config: {dino_config}")

    # To save the config locally:
    dino_config.save_pretrained("Dino_v3.json")

    dino_config.architectures = ["DinoV3WithAdapterBackbone"]

    # Now you can inspect, modify, and save it:
    print(f"print Dino model config with custom config: {dino_config}")

    model.config.backbone_config = dino_config

    return model
def get_model_info(model: AutoModelForUniversalSegmentation) -> Dict:
    """
    Get information about the DINOv3-Mask2Former model.

    Args:
        model: The DINOv3-Mask2Former model

    Returns:
        Dictionary with model information
    """
    backbone = model.model.backbone

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in backbone.model.parameters())
    frozen_params = sum(
        p.numel() for p in backbone.model.parameters() if not p.requires_grad
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbone_parameters": backbone_params,
        "frozen_parameters": frozen_params,
        "backbone_model": backbone.model.config.name_or_path
        if hasattr(backbone.model.config, "name_or_path")
        else "DINOv3",
        "output_channels": list(backbone._out_feature_channels.values()),
        "output_strides": list(backbone._out_feature_strides.values()),
    }
