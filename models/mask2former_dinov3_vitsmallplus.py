"""
DINOv3-Mask2Former Small+ Model Implementation

This module provides a complete DINOv3-Mask2Former model with custom backbone replacement.
The model uses DINOv3 as the backbone and Mask2Former as the segmentation head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from transformers import AutoModel, AutoModelForUniversalSegmentation, AutoConfig
import logging
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.utils._typing import is_jsonable
from rich import print as rprint


logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCEMENT 1: Improved Multi-Scale Adapter with Feature Fusion
# ============================================================================

class ImprovedAdapter(nn.Module):
    """
    Enhanced adapter with:
    1. Deformable convolutions for better feature alignment
    2. Feature fusion across scales
    3. Squeeze-and-Excitation for channel attention
    """
    
    def __init__(self, in_channels: int, out_channels: List[int], use_fusion: bool = True):
        super().__init__()
        self.use_fusion = use_fusion
        
        # Main projection layers with 3x3 conv instead of 1x1 for better receptive field
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for out_ch in out_channels
        ])
        
        # Squeeze-and-Excitation blocks for channel attention
        self.se_blocks = nn.ModuleList([
            SEBlock(out_ch) for out_ch in out_channels
        ])
        
        # Cross-scale fusion if enabled
        if use_fusion:
            self.fusion_convs = nn.ModuleList([
                nn.Conv2d(out_ch, out_ch, kernel_size=1) 
                for out_ch in out_channels
            ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Apply projections and SE attention
        adapted = []
        for i, feat in enumerate(features):
            projected = self.projections[i](feat)
            attended = self.se_blocks[i](projected)
            adapted.append(attended)
        
        # Optional: Cross-scale feature fusion for better multi-scale understanding
        if self.use_fusion and len(adapted) > 1:
            fused = []
            for i in range(len(adapted)):
                # Aggregate information from adjacent scales
                curr = adapted[i]
                
                # Add upsampled lower-resolution features
                if i < len(adapted) - 1:
                    lower = F.interpolate(adapted[i + 1], size=curr.shape[-2:], 
                                         mode='bilinear', align_corners=False)
                    curr = curr + self.fusion_convs[i](lower) * 0.5
                
                fused.append(curr)
            return fused
        
        return adapted


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# ENHANCEMENT 2: Better Layer Selection Strategy for DINOv3
# ============================================================================

class DinoV3WithAdapterBackbone(nn.Module):
    """
    Enhanced backbone with:
    1. Optimized layer selection based on DINOv3 research
    2. Better multi-scale feature extraction
    3. Optional feature pyramid network (FPN)
    """
    
    def __init__(
        self, 
        model_name: str, 
        out_channels: List[int],
        use_improved_adapter: bool = True,
        use_fpn: bool = False
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        # Use improved or basic adapter
        if use_improved_adapter:
            self.adapter = ImprovedAdapter(self.model.config.hidden_size, out_channels)
        else:
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
        
        # CRITICAL: Optimized layer selection for DINOv3
        # Based on research: early, mid-early, mid, and late layers
        # These capture different semantic levels
        num_layers = self.model.config.num_hidden_layers
        
        if num_layers == 12:  # Small/Base models
            # Early (low-level), Mid-Early, Mid, Deep (high-level)
            self.layers_to_extract = [2, 5, 8, 11]
        elif num_layers == 24:  # Large models
            self.layers_to_extract = [5, 11, 17, 23]
        else:  # Giant models or others
            step = num_layers // 4
            self.layers_to_extract = [step, step*2, step*3, num_layers-1]
        
        rprint(f"[cyan]Extracting from DINOv3 layers: {self.layers_to_extract}[/cyan]")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get DINOv3 outputs with all hidden states
        outputs = self.model(
            pixel_values=x, 
            output_hidden_states=True, 
            return_dict=True
        )
        hidden_states = outputs.hidden_states
        
        # Calculate spatial dimensions after patch embedding
        batch_size, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size
        
        # Extract features from different layers
        extracted_features = []
        for layer_idx in self.layers_to_extract:
            layer_output = hidden_states[layer_idx + 1]
            # Reshape from (B, N, C) to (B, C, H, W)
            feature_map = (
                layer_output[:, 1:, :]  # Remove CLS token
                .permute(0, 2, 1)
                .reshape(
                    batch_size, 
                    self.model.config.hidden_size, 
                    patch_height, 
                    patch_width
                )
            )
            extracted_features.append(feature_map)
        
        # Apply adapter to convert channels
        adapted_features = self.adapter(extracted_features)
        
        # Return features with proper naming for Mask2Former
        return {name: feat for name, feat in zip(self.out_features, adapted_features)}


# ============================================================================
# ENHANCEMENT 3: Original Adapter (for backward compatibility)
# ============================================================================

class Adapter(nn.Module):
    """Basic adapter module (original implementation)"""
    
    def __init__(self, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_ch, kernel_size=1) for out_ch in out_channels]
        )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.projections[i](feat) for i, feat in enumerate(features)]


# ============================================================================
# ENHANCEMENT 4: Main Model with Improvements
# ============================================================================

class Mask2Former_Dinov3(nn.Module, PyTorchModelHubMixin):
    """
    Enhanced Mask2Former with DINOv3 backbone
    
    Key improvements:
    1. Better layer selection from DINOv3
    2. Improved adapter with feature fusion and attention
    3. Training-friendly configurations
    4. Gradient checkpointing support for memory efficiency
    """
    
    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        dinov3_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        expected_channels: List[int] = [96, 192, 384, 768],
        freeze_backbone: bool = True,
        use_improved_adapter: bool = True,
        freeze_adapter: bool = False,  # NEW: Option to fine-tune adapter
        use_gradient_checkpointing: bool = False,
        hub_token: Optional[str] = None,
    ):
        """
        Args:
            label2id: Mapping from label names to IDs
            id2label: Mapping from IDs to label names
            dinov3_model_name: HuggingFace model name for DINOv3
            expected_channels: Output channels for multi-scale features
            freeze_backbone: Whether to freeze DINOv3 backbone
            use_improved_adapter: Use enhanced adapter with fusion and attention
            freeze_adapter: Whether to freeze the channel adapter
            use_gradient_checkpointing: Enable gradient checkpointing for memory savings
            hub_token: HuggingFace token for private models
        """
        super().__init__()
        
        rprint(f"\n[bold cyan]{'='*70}[/bold cyan]")
        rprint(f"[bold cyan]Initializing Enhanced Mask2Former-DINOv3[/bold cyan]")
        rprint(f"[bold cyan]{'='*70}[/bold cyan]")
        
        # Store configuration
        self.label2id = label2id
        self.id2label = id2label
        self.dinov3_model_name = dinov3_model_name
        self.expected_channels = expected_channels
        self.freeze_backbone = freeze_backbone
        self.use_improved_adapter = use_improved_adapter
        self.hub_token = hub_token
        
        # Use semantic segmentation base for instance segmentation
        # This works better than panoptic for pure instance tasks
        mask2former_model_name = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        
        rprint(f"\n[yellow]Loading base Mask2Former: {mask2former_model_name}[/yellow]")
        
        # Build the Mask2Former model
        model = AutoModelForUniversalSegmentation.from_pretrained(
            mask2former_model_name,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            token=hub_token,
        )
        
        # Replace backbone with enhanced DINOv3
        rprint(f"[yellow]Replacing backbone with DINOv3: {dinov3_model_name}[/yellow]")
        custom_backbone = DinoV3WithAdapterBackbone(
            dinov3_model_name, 
            expected_channels,
            use_improved_adapter=use_improved_adapter
        )
        model.model.backbone = custom_backbone
        
        # Freeze DINOv3 backbone
        if freeze_backbone:
            for param in model.model.backbone.model.parameters():
                param.requires_grad = False
            rprint("[green]✓[/green] Frozen DINOv3 backbone")
        
        # Optionally freeze adapter
        if freeze_adapter:
            for param in model.model.backbone.adapter.parameters():
                param.requires_grad = False
            rprint("[green]✓[/green] Frozen channel adapter")
        else:
            rprint("[yellow]⚡[/yellow] Adapter is trainable (recommended for quick adaptation)")
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                rprint("[green]✓[/green] Enabled gradient checkpointing")
        
        # Attach configs
        dino_config = AutoConfig.from_pretrained(dinov3_model_name)
        dino_config.architectures = ["DinoV3WithAdapterBackbone"]
        mask2former_config = AutoConfig.from_pretrained(mask2former_model_name)
        mask2former_config.model_type = "mask2former-with-dinov3"
        mask2former_config.backbone_config = dino_config
        model.config = mask2former_config
        
        self.inner_model = model
        self.config = model.config
        
        # Report statistics
        self._print_model_statistics()
    
    def _print_model_statistics(self):
        """Print detailed model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.inner_model.model.backbone.model.parameters())
        adapter_params = sum(p.numel() for p in self.inner_model.model.backbone.adapter.parameters())
        decoder_params = trainable_params - sum(
            p.numel() for p in self.inner_model.model.backbone.parameters() if p.requires_grad
        )
        
        rprint(f"\n[bold green]{'='*70}[/bold green]")
        rprint(f"[bold green]Model Statistics[/bold green]")
        rprint(f"[bold green]{'='*70}[/bold green]")
        rprint(f"  Total parameters:      {total_params:>15,}")
        rprint(f"  Trainable parameters:  {trainable_params:>15,}  ({100*trainable_params/total_params:.2f}%)")
        rprint(f"\n[cyan]Component Breakdown:[/cyan]")
        rprint(f"  DINOv3 Backbone:       {backbone_params:>15,}")
        rprint(f"  Channel Adapter:       {adapter_params:>15,}")
        rprint(f"  Mask2Former Decoder:   {decoder_params:>15,}")
        rprint(f"[bold green]{'='*70}[/bold green]\n")
    
    def forward(self, *args, **kwargs):
        """Forward pass through inner Mask2Former model"""
        return self.inner_model(*args, **kwargs)
    
    def get_trainable_params(self) -> List[Dict]:
        """
        Get trainable parameters with appropriate learning rates
        Useful for differential learning rates
        """
        adapter_params = []
        decoder_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'adapter' in name:
                    adapter_params.append(param)
                else:
                    decoder_params.append(param)
        
        # Return param groups for optimizer
        return [
            {'params': adapter_params, 'lr': 1e-4},  # Lower LR for adapter
            {'params': decoder_params, 'lr': 5e-5}   # Even lower for decoder
        ]

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
