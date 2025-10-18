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
# from huggingface_hub import PyTorchModelHubMixin

logger = logging.getLogger(__name__)
from huggingface_hub import PyTorchModelHubMixin


class Adapter(nn.Module):
    """
    Adapter module to convert DINOv3 features to expected channels for Mask2Former head.
    """

    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.projections[i](feat) for i, feat in enumerate(features)]


class DinoV3WithAdapterBackbone(nn.Module):
    """
    Custom backbone that combines DINOv3 with adapter layers for Mask2Former compatibility.
    """

    def __init__(self, model_name: str, out_channels: List[int]):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.adapter = Adapter(self.model.config.hidden_size, out_channels)

        # Define output features for Mask2Former compatibility
        self.out_features = [f"stage_{i}" for i in range(len(out_channels))]
        self._out_feature_channels = {
            name: ch for name, ch in zip(self.out_features, out_channels)
        }
        self._out_feature_strides = {
            "stage_0": 8,
            "stage_1": 16,
            "stage_2": 32,
            "stage_3": 32,
        }

        # Layers to extract from DINOv3 (different depths for multi-scale features)
        self.layers_to_extract = [2, 5, 8, 11]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get DINOv3 outputs with all hidden states
        outputs = self.model(
            pixel_values=x, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states

        # Calculate spatial dimensions after patch embedding
        batch_size, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size

        # Extract features from different layers
        extracted_features = []
        for layer_idx in self.layers_to_extract:
            layer_output = hidden_states[layer_idx + 1]  # Skip CLS token
            # Reshape from (B, N, C) to (B, C, H, W)
            feature_map = (
                layer_output[:, 1:, :]
                .permute(0, 2, 1)
                .reshape(
                    batch_size, self.model.config.hidden_size, patch_height, patch_width
                )
            )
            extracted_features.append(feature_map)

        # Apply adapter to convert channels
        adapted_features = self.adapter(extracted_features)

        # Return features with proper naming for Mask2Former
        return {name: feat for name, feat in zip(self.out_features, adapted_features)}

from huggingface_hub.utils._typing import is_jsonable

class Mask2Former_Dinov3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        dinov3_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        expected_channels: List[int] = [96, 192, 384, 768],
        freeze_backbone: bool = True,
        hub_token: str = None,
        # mask2former_model_name: str = "facebook/mask2former-swin-small-coco-instance"
    ):
        super().__init__()

        print(f"is_jsonable label2id: {is_jsonable(label2id)}")

        self.label2id = label2id
        self.id2label = id2label
        self.dinov3_model_name = dinov3_model_name
        self.expected_channels = expected_channels
        self.freeze_backbone = freeze_backbone
        self.hub_token = hub_token
        # self.mask2former_model_name = mask2former_model_name

        mask2former_model_name = "facebook/mask2former-swin-small-coco-instance"
        # Build the Mask2Former model
        model = AutoModelForUniversalSegmentation.from_pretrained(
            mask2former_model_name,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            token=hub_token,
        )
        custom_backbone = DinoV3WithAdapterBackbone(
            dinov3_model_name, expected_channels
        )
        model.model.backbone = custom_backbone
        if freeze_backbone:
            for param in model.model.backbone.model.parameters():
                param.requires_grad = False

        # Attach configs
        dino_config = AutoConfig.from_pretrained(dinov3_model_name)
        dino_config.architectures = ["DinoV3WithAdapterBackbone"]
        mask2former_model_name_config = AutoConfig.from_pretrained(mask2former_model_name)
        mask2former_model_name_config.model_type = "mask2former-with-dinov3"
        mask2former_model_name_config.backbone_config = dino_config
        model.config = mask2former_model_name_config

        self.inner_model = model
        self.config = model.config  # For HF compatibility

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


# class Mask2Former_Dinov3(
#     nn.Module,
#     PyTorchModelHubMixin,
#     library_name="Mask2Former_Dinov3",
#     tags=["Mask2Former", "Dino_v3"],
# ):
#     def __init__(
#         self,
#     ):
        
#         super().__init__()


#         self.config = None

#         self.inner_model = None


#     def create_mask2former_dinov3_model(
#         self,
#         label2id: Dict[str, int],
#         id2label: Dict[int, str],
#         dinov3_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
#         expected_channels: List[int] = [96, 192, 384, 768],
#         freeze_backbone: bool = True,
#         hub_token: str = None,
#     ) -> AutoModelForUniversalSegmentation:

    
#         """
#         Create a complete DINOv3-Mask2Former model with custom backbone replacement.
#         Args:
#             label2id: Dictionary mapping label names to IDs
#             id2label: Dictionary mapping IDs to label names
#             dinov3_model_name: HuggingFace model name for DINOv3
#             expected_channels: List of output channels for each stage
#             freeze_backbone: Whether to freeze DINOv3 backbone weights
#             hub_token: HuggingFace Hub token if needed

#         Returns:
#             Complete DINOv3-Mask2Former model ready for training/inference
#         """
#         # Fixed Mask2Former base model
#         mask2former_model_name = "facebook/mask2former-swin-small-coco-instance"

#         logger.info(f"Creating DINOv3-Mask2Former model...")
#         logger.info(f"  - Mask2Former base: {mask2former_model_name}")
#         # logger.info(f"  - DINOv3 backbone: {dinov3_model_name}")
#         # logger.info(f"  - Expected channels: {expected_channels}")
#         logger.info(f"  - Freeze backbone: {freeze_backbone}")

#         # TODO: The model here loaded into the cpu, check it!
#         # TODO: Why we use AutoModelForUniversalSegmentation instead of AutoModelForInstanceSegmentation
#         # 1. Load the base Mask2Former model
#         model = AutoModelForUniversalSegmentation.from_pretrained(
#             mask2former_model_name,
#             label2id=label2id,
#             id2label=id2label,
#             ignore_mismatched_sizes=True,
#             token=hub_token,
#         )

#         # 2. Create custom DINOv3 backbone with adapter
#         custom_backbone = DinoV3WithAdapterBackbone(
#             dinov3_model_name, expected_channels
#         )

#         # 3. Replace the backbone
#         model.model.backbone = custom_backbone

#         # 4. Freeze DINOv3 weights if requested
#         if freeze_backbone:
#             for param in model.model.backbone.model.parameters():
#                 param.requires_grad = False
#             logger.info("DINOv3 backbone weights frozen.")
#         else:
#             logger.info("DINOv3 backbone weights remain trainable.")

#         logger.info("Successfully created DINOv3-Mask2Former model.")

#         # 5. update the model configuration as well:

#         # model.config.backbone = "dinov3"
#         # model.config.model_type = "mask2former-with-dinov3"
#         # Optionally, add more info about your adapter design, layer indices, etc.

#         # Download the config automatically from Hugging Face
#         dino_config = AutoConfig.from_pretrained(dinov3_model_name)

#         mask2former_model_name_config = AutoConfig.from_pretrained(mask2former_model_name)

#         mask2former_model_name_config.model_type = "mask2former-with-dinov3"

#         # Now you can inspect, modify, and save it:
#         print(f"print Dino model config: {dino_config}")

#         # To save the config locally:
#         # dino_config.save_pretrained("Dino_v3.json")

#         dino_config.architectures = ["DinoV3WithAdapterBackbone"]

#         # Now you can inspect, modify, and save it:
#         print(f"print Dino model config with custom config: {dino_config}")

#         # model.config.backbone_config = dino_config

#         mask2former_model_name_config.backbone_config = dino_config

        
#         # self.dino_config_dict = dino_config_dict

#         # self.mask2former_config_dict = model.config
        
#         model.config = mask2former_model_name_config

#         self.inner_model = model

#         self.config = model.config

#         return self.inner_model
    
    # def forward(self, *args, **kwargs):
    #     return self.inner_model(*args, **kwargs)


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
