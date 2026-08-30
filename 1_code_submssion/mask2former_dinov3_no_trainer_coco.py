#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 [Chan Ho Bae / GitHub @Carti-97]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is a modification of the Hugging Face Transformers instance segmentation example.
It has been adapted to replace the original backbone with DINOv3 models and support the COCO dataset format.
"""

"""Finetuning 🤗 Transformers model for instance segmentation with Accelerate 🚀."""

import argparse
import json
import logging
import math
import os
import sys
import pickle
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import albumentations as A
import datasets
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection import IntersectionOverUnion
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image

import transformers
from transformers import (
    AutoImageProcessor,
    SchedulerType,
    get_scheduler,
)
from transformers.image_processing_utils import BatchFeature
from transformers.utils import check_min_version

from transformers.utils.versions import require_version

import importlib.util

from models.mask2former_dinov3_vitsmallplus import Mask2Former_Dinov3

import glob

import torch.nn.functional as F


logger = logging.getLogger(__name__)

from rich import traceback, pretty
traceback.install(show_locals=True)

pretty.install()
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.56.0.dev0")

from rich import print as rprint


import torch

def get_parameter_groups(model, base_lr=5e-5):
    """
    Create parameter groups with differential learning rates
    
    Strategy:
    - Adapter: Highest LR (needs most adaptation)
    - Encoder: Medium LR (some adaptation needed)
    - Decoder: Lower LR (already pretrained on Mapillary)
    - Class/Mask heads: High LR (task-specific)
    """
    
    # Initialize parameter groups
    adapter_params = []
    encoder_params = []
    decoder_params = []
    head_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Group 1: Channel adapter (highest LR - needs most learning)
        if 'adapter' in name.lower() or 'backbone.adapter' in name:
            adapter_params.append(param)
            
        # Group 2: Transformer encoder (medium LR)
        elif 'encoder' in name.lower() or 'input_proj' in name or 'level_embed' in name:
            encoder_params.append(param)
            
        # Group 3: Prediction heads (high LR - task-specific)
        elif 'class_predictor' in name or 'mask_predictor' in name or 'class_embed' in name or 'mask_embed' in name:
            head_params.append(param)
            
        # Group 4: Decoder (lower LR - already well-pretrained)
        elif 'decoder' in name.lower() or 'transformer_module' in name:
            decoder_params.append(param)
            
        # Group 5: Everything else (medium LR)
        else:
            other_params.append(param)
    
    # Create parameter groups with differential LRs
    param_groups = []
    
    if adapter_params:
        param_groups.append({
            'params': adapter_params,
            'lr': base_lr * 10,  # 10x base LR for adapter
            'name': 'adapter'
        })
        print(f"✓ Adapter params: {sum(p.numel() for p in adapter_params):,} with LR={base_lr * 10}")
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr * 5,  # 5x base LR for prediction heads
            'name': 'heads'
        })
        print(f"✓ Head params: {sum(p.numel() for p in head_params):,} with LR={base_lr * 5}")
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': base_lr * 2,  # 2x base LR for encoder
            'name': 'encoder'
        })
        print(f"✓ Encoder params: {sum(p.numel() for p in encoder_params):,} with LR={base_lr * 2}")
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr,  # Base LR for decoder (already trained)
            'name': 'decoder'
        })
        print(f"✓ Decoder params: {sum(p.numel() for p in decoder_params):,} with LR={base_lr}")
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr * 2,  # 2x base LR for other components
            'name': 'other'
        })
        print(f"✓ Other params: {sum(p.numel() for p in other_params):,} with LR={base_lr * 2}")
    
    return param_groups


# Replace your optimizer initialization with this:
def create_optimizer(model, args):
    """
    Create AdamW optimizer with differential learning rates
    """
    # Get parameter groups with differential LRs
    # Base LR from args, components get multipliers
    param_groups = get_parameter_groups(model, base_lr=args.learning_rate)
    
    # Create optimizer with parameter groups
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.05,
        betas=[args.adam_beta1, args.adam_beta2] if hasattr(args, 'adam_beta1') else [0.9, 0.999],
        eps=args.adam_epsilon if hasattr(args, 'adam_epsilon') else 1e-8,
    )
    
    print(f"\n{'='*70}")
    print("Optimizer Configuration:")
    print(f"{'='*70}")
    print(f"Base Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {0.05}")
    print(f"Betas: {[0.9, 0.999]}")
    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*70}\n")
    
    return optimizer


# Usage in your training script:
# Replace:
# optimizer = torch.optim.AdamW(
#     list(model.parameters()),
#     ...
# )

# With:





def load_model_from_config(model_path: str):
    """
    Dynamically load model creation function and mask2former model name from the specified Python file.
    
    Args:
        model_path: Path to the Python model file (e.g., "models/mask2former_dinov3_vitsmallplus.py")
        
    Returns:
        Tuple of (create_mask2former_dinov3_model function, mask2former_model_name string)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Extract module name from file path
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Load module dynamically
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Get the model creation function
    if not hasattr(module, 'create_mask2former_dinov3_model'):
        raise AttributeError(f"Model file {model_path} does not contain 'create_mask2former_dinov3_model' function")
    
    # Extract mask2former_model_name from the function source by executing it partially
    # This is a bit hacky but works - we'll look at the function's source code
    import inspect
    func_source = inspect.getsource(module.create_mask2former_dinov3_model)
    
    # Extract mask2former_model_name from function source
    mask2former_model_name = None
    for line in func_source.split('\n'):
        line = line.strip()
        if line.startswith('mask2former_model_name') and '=' in line:
            # Parse the line: mask2former_model_name = "facebook/mask2former-swin-small-coco-instance"
            mask2former_model_name = line.split('=', 1)[1].strip().strip('"\'')
            break
    
    if not mask2former_model_name:
        logger.warning(f"Could not find mask2former_model_name in {model_path}, using default")
        mask2former_model_name = "facebook/mask2former-swin-small-coco-instance"
    
    logger.info(f"Successfully loaded model from: {model_path}")
    logger.info(f"  - Detected mask2former base model: {mask2former_model_name}")
    
    return module.create_mask2former_dinov3_model, mask2former_model_name

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/instance-segmentation/requirements.txt")


class MapillaryInstanceDataset(Dataset):
    """
    Mapillary Vistas Dataset for Instance Segmentation
    """
    def __init__(self, root_dir, image_processor, version='v2.0', split='training', transforms=None):
        """
        Args:
            root_dir: Root directory containing the dataset
            version: Dataset version ('v1.2' or 'v2.0')
            split: 'training' or 'validation'
            transforms: Transformations to apply
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.version = version
        self.split = split
        # self.split = 'validation'
        self.transforms = transforms

        # Load config file to get label information
        config_path = os.path.join(root_dir, f'config_{version}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.labels = config['labels']

        # Get only "thing" classes (classes with instances)
        self.thing_classes = [label for label in self.labels if label["instances"]]
        self.label_id_to_class_id = {}

        # Create mapping from original label_id to class_id (1-indexed for PyTorch)
        # Class 0 is reserved for background
        self.categories = {}
        for class_id, label in enumerate(self.labels):
            if label["instances"]:
                # +1 because 0 is background
                self.label_id_to_class_id[class_id] = len([l for l in self.labels[:class_id+1] if l["instances"]])
                self.categories[self.label_id_to_class_id[class_id]] = label["readable"]

        # Get all image IDs
        images_dir = os.path.join(root_dir, split, 'images')
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]


        if split == "training":

            N_train = 3000
           
            self.image_ids = self.image_ids[:N_train]  # Only use first N
            # self.image_ids = ['--BJs76vloEaiH-wppzWNA']
            self.image_files = self.image_files[:N_train]
            print(f"self.image_files: {self.image_files}")
            with open("image_files.txt", "w") as f:
                for file_path in self.image_files:
                    f.write(file_path + "\n")

            print("Saved image file list to image_files.txt")

        else:
            N_val = 600

            self.image_ids = self.image_ids[:N_val]  # Only use first N
            # self.image_ids = ['--BJs76vloEaiH-wppzWNA']
            self.image_files = self.image_files[:N_val]

            print(f"self.image_files: {self.image_files}")
            with open("image_files_valid.txt", "w") as f:
                for file_path in self.image_files:
                    f.write(file_path + "\n")

        print(f"Loaded {len(self.image_ids)} images from {split} split")
        print(f"Number of thing classes (with instances): {len(self.thing_classes)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.root_dir, self.split, 'images', f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        h, w = image.height, image.width

        # Load instance image
        instance_path = os.path.join(self.root_dir, self.split, self.version, 'instances', f'{image_id}.png')
        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)

        # Create instance mask with only "thing" classes
        instance_mask = np.zeros_like(instance_array, dtype=np.uint16)

        instance_to_semantic = {}

        inst_id_counter = 1
        for inst_value in np.unique(instance_array):
            if inst_value == 0:
                continue
            label_id = inst_value // 256
            # Only keep "thing" classes
            if not self.labels[label_id]["instances"]:
                continue
            # Assign contiguous instance IDs (1, 2, 3, ...)
            instance_mask[instance_array == inst_value] = inst_id_counter
            # Map back to its semantic class
            class_id = self.label_id_to_class_id.get(label_id, 0)
            instance_to_semantic[inst_id_counter] = class_id
            inst_id_counter += 1

        # Save mapping for use after transforms
        orig_instance_to_semantic = instance_to_semantic.copy()

        # Apply transforms if provided
        if self.transforms is not None:
            image_np = np.array(image)
            output = self.transforms(image=image_np, mask=instance_mask)
            image = output["image"]
            instance_mask = output["mask"]

            unique_ids = np.unique(instance_mask)
            # Only keep instance IDs that existed before transforms
            instance_to_semantic = {
                int(inst_id): orig_instance_to_semantic[int(inst_id)]
                for inst_id in unique_ids
                if inst_id > 0 and int(inst_id) in orig_instance_to_semantic
            }
        else:
            image = np.array(image)

        # Remove background key from mapping if present (robust fix)
        if 0 in instance_to_semantic:
            del instance_to_semantic[0]
        assert 0 not in instance_to_semantic, "Instance mapping incorrectly includes background!"

        # Apply image processor (final step)
        instance_mask = instance_mask.astype(np.int32) 
        inputs = self.image_processor(
            images=[image],
            segmentation_maps=[instance_mask],
            instance_id_to_semantic_id=instance_to_semantic,
            return_tensors="pt",
        )
        return {
            "pixel_values": inputs.pixel_values[0],
            "mask_labels": inputs.mask_labels[0],
            "class_labels": inputs.class_labels[0],
            "original_size": (h, w),
        }
    


    def get_num_classes(self):
        """Returns number of classes including background"""
        return len(self.thing_classes) + 1  # +1 for background



def augment_and_transform_batch(
    examples: Mapping[str, Any], transform: A.Compose, image_processor: AutoImageProcessor
) -> BatchFeature:
    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }

    for pil_image, pil_annotation in zip(examples["image"], examples["annotation"]):
        image = np.array(pil_image)
        semantic_and_instance_masks = np.array(pil_annotation)[..., :2]

        output = transform(image=image, mask=semantic_and_instance_masks)
        aug_image = output["image"]
        aug_semantic_and_instance_masks = output["mask"]
        aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

        # Mapping for foreground only
        unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
        instance_id_to_semantic_id = {
            int(instance_id): int(semantic_id)
            for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
            if instance_id > 0
        }
        assert 0 not in instance_id_to_semantic_id, "Background (instance_id=0) included in mapping!"

        model_inputs = image_processor(
            images=[aug_image],
            segmentation_maps=[aug_instance_mask],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )

        batch["pixel_values"].append(model_inputs.pixel_values[0])
        batch["mask_labels"].append(model_inputs.mask_labels[0])
        batch["class_labels"].append(model_inputs.class_labels[0])

    return batch


def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    
    # ================== FIX START ==================
    if "original_size" in examples[0]:
        batch["original_sizes"] = [example["original_size"] for example in examples]
    # ================== FIX END ==================
    
    return batch


def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors


def evaluation_loop(model, image_processor, accelerator: Accelerator, dataloader, id2label):
    # Each GPU maintains its own metric instance
    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True).to(accelerator.device)

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not accelerator.is_local_main_process):
        original_sizes = inputs.pop("original_sizes", None)
        
        try:
            
            with torch.no_grad():
                outputs = model(**inputs)

                
            # Get target sizes for current batch
            if original_sizes is not None:
                current_target_sizes = original_sizes
            else:
                current_target_sizes = [masks.shape[-2:] for masks in inputs["mask_labels"]]
            
            # Process predictions for current batch (local GPU only)
            post_processed_output = image_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                target_sizes=current_target_sizes,
                return_binary_maps=True,
            )
        except Exception as e:
            rprint(f"An error occurred: {e}")
            continue
        # Prepare predictions and targets for current batch
        post_processed_predictions = []
        post_processed_targets = []
        
        for idx, (image_predictions, target_size) in enumerate(zip(post_processed_output, current_target_sizes)):
            # Prediction
            if image_predictions["segments_info"]:
                pred = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]], 
                                          device=accelerator.device),
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]], 
                                          device=accelerator.device),
                }
            else:
                pred = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool, device=accelerator.device),
                    "labels": torch.tensor([], device=accelerator.device),
                    "scores": torch.tensor([], device=accelerator.device),
                }
            post_processed_predictions.append(pred)
            
            # Target (upsample mask to original size if needed)
            target_masks = inputs["mask_labels"][idx].to(dtype=torch.bool)

            if target_masks.shape[-2:] != tuple(target_size):
                # Upsample all masks [N, H, W] to [N, target_H, target_W]
                target_masks = F.interpolate(
                    target_masks.float().unsqueeze(0), size=tuple(target_size), mode='bi'
                )[0].bool()


            # Target
            target = {
                "masks": target_masks,
                "labels": inputs["class_labels"][idx],
            }

            post_processed_targets.append(target)

            # import matplotlib.pyplot as plt
            # import numpy as np

            # def plot_pred_target_masks(pred, target, index=0):
            #     # pred['masks']: [num_pred, H, W] bool
            #     # target['masks']: [num_target, H, W] bool

            #     if pred['masks'].shape[0] == 0 or target['masks'].shape[0] == 0:
            #         print("No masks to plot for this prediction/target.")
            #         return

            #     # For demonstration: overlay first predicted mask and first target mask
            #     pred_mask = pred['masks'][0].cpu().numpy()
            #     target_mask = target['masks'][0].cpu().numpy()
                
            #     # Optional: create a semi-transparent overlay for comparison
            #     overlay = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
            #     overlay[..., 0] = pred_mask.astype(np.float32)  # Red for prediction
            #     overlay[..., 1] = target_mask.astype(np.float32)  # Green for target
            #     overlay[..., 2] = 0  # Blue channel empty

            #     plt.figure(figsize=(8,8))
            #     plt.imshow(overlay, alpha=0.7)
            #     plt.title(f"Overlay: prediction (red), target (green)")
            #     plt.axis('off')
            #     plt.show()

            # # In evaluation loop, after creating post_processed_predictions/targets:
            # for idx, (pred, target) in enumerate(zip(post_processed_predictions, post_processed_targets)):
            #     plot_pred_target_masks(pred, target, index=idx)
            #     # Optional: break or limit the number plotted per batch for efficiency
            #     if idx > 5: break


        # import matplotlib.pyplot as plt
        # import numpy as np
        # for target in post_processed_targets:
        #     # target["masks"]: [num_instances, H, W]
        #     # If you produce a combined mask map per pixel (class ids)
        #     target_mask_map = torch.zeros_like(target['masks'][0], dtype=torch.int64)
        #     for mask, label in zip(target['masks'], target['labels']):
        #         target_mask_map[mask] = label
        #     # Now check for background
        #     has_background = (target_mask_map == 0).any().item()
        #     if has_background:
        #         print("Background pixels detected in this mask.")

        # combined_pred = np.sum(pred['masks'].cpu().numpy(), axis=0)
        # combined_target = np.sum(target['masks'].cpu().numpy(), axis=0)
        # overlay = np.zeros((*combined_pred.shape, 3), dtype=np.float32)
        # overlay[..., 0] = (combined_pred > 0)  # Red = any pred mask pixel
        # overlay[..., 1] = (combined_target > 0)  # Green = any target mask pixel

        # plt.imshow(overlay, alpha=0.7)
        # plt.title("All Predicted (Red) vs Target Masks (Green)")
        # plt.show()

        # rprint(f"post_processed_predictions: {post_processed_predictions}")
        # rprint(f"post_processed_targets: {post_processed_targets}")

        # Update metric locally (no gathering)
        metric.update(post_processed_predictions, post_processed_targets)

    results = metric.compute()
    return results


def setup_logging(accelerator: Accelerator) -> None:
    """Setup logging according to `training_args`."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        logger.setLevel(logging.INFO)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


import argparse
import json
import os
from transformers import SchedulerType



def handle_repository_creation(accelerator: Accelerator, args: argparse.Namespace):
    """Create a repository for the model and dataset if `args.push_to_hub` is set."""

    repo_id = None
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    return repo_id



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model for instance segmentation task")

    # JSON config file option (at the beginning)
    parser.add_argument(
        "--config",
        type=str,
        default="mask2former-dinov3_smallplus_1024_train_args.json",
        help="Path to JSON config file containing training arguments"
    )

    # Temporarily parse only the config argument first.
    temp_args, _ = parser.parse_known_args()

    # A dictionary containing default values
    defaults = {}
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            defaults = json.load(f)

    # ========================================================================
    # Model and Dataset Configuration
    # ========================================================================
    parser.add_argument(
        "--model",
        type=str,
        help="Path to a pretrained model or model identifier from huggingface.co/models.",
        default="models/mask2former_dinov3_vitsmallplus.py",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset on the hub or path to mapillary dataset.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    
    # ========================================================================
    # Image Processing Configuration
    # ========================================================================
    parser.add_argument(
        "--image_height",
        type=int,
        default=512,  # Changed from 384 to 512 for better quality
        help="The height of the images to feed the model.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=512,  # Changed from 384 to 512 for better quality
        help="The width of the images to feed the model.",
    )
    parser.add_argument(
        "--do_reduce_labels",
        action="store_true",
        help="Whether to reduce the number of labels by removing the background class.",
    )
    
    # ========================================================================
    # Data Loading Configuration
    # ========================================================================
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path to a folder in which the model and dataset will be cached.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,  # Changed from 8 to 4 for M3 Max (safer with 512x512)
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,  # Changed from 8 to 4
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,  # Changed from 0 to 4 for faster data loading
        help="Number of workers to use for the dataloaders.",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,  # NEW: Prefetch batches for speed
        help="Number of batches to prefetch per worker.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        default=False,  # False for MPS (True for CUDA)
        help="Whether to pin memory in data loaders.",
    )
    
    # ========================================================================
    # Optimization Configuration
    # ========================================================================
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,  # Base learning rate for differential LR
        help="Base learning rate (will be modified for different components with differential LR).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,  # NEW: Increased for better generalization
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,  # NEW: Gradient clipping
        help="Maximum gradient norm for clipping.",
    )
    
    # ========================================================================
    # NEW: Differential Learning Rate Configuration
    # ========================================================================
    parser.add_argument(
        "--use_differential_lr",
        action="store_true",
        default=True,  # Enable by default
        help="Whether to use differential learning rates for different model components.",
    )
    parser.add_argument(
        "--adapter_lr_multiplier",
        type=float,
        default=10.0,  # Adapter gets 10x base LR
        help="Learning rate multiplier for adapter layers.",
    )
    parser.add_argument(
        "--head_lr_multiplier",
        type=float,
        default=5.0,  # Prediction heads get 5x base LR
        help="Learning rate multiplier for classification/mask prediction heads.",
    )
    parser.add_argument(
        "--encoder_lr_multiplier",
        type=float,
        default=2.0,  # Encoder gets 2x base LR
        help="Learning rate multiplier for transformer encoder.",
    )
    parser.add_argument(
        "--decoder_lr_multiplier",
        type=float,
        default=1.0,  # Decoder gets base LR (already pretrained)
        help="Learning rate multiplier for transformer decoder.",
    )
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=25,  # Increased from 20 to 25 for better convergence
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,  # Effective batch size = 4 * 4 = 16
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    # ========================================================================
    # Learning Rate Scheduler Configuration
    # ========================================================================
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",  # Changed from linear to cosine
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=None,  # Will be calculated from warmup_ratio
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,  # NEW: 5% warmup (more flexible than fixed steps)
        help="Ratio of total training steps to use for warmup.",
    )
    parser.add_argument(
        "--lr_end_ratio",
        type=float,
        default=0.01,  # NEW: End at 1% of initial LR
        help="Ratio of initial learning rate to end at (for cosine/polynomial schedulers).",
    )
    
    # ========================================================================
    # Evaluation and Checkpointing
    # ========================================================================
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="500",  # Save every 500 steps
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",  # NEW: Evaluate on steps
        choices=["no", "steps", "epoch"],
        help="The evaluation strategy to use.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,  # NEW: Evaluate every 500 steps
        help="Number of update steps between evaluations.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",  # NEW: Save on steps
        choices=["no", "steps", "epoch"],
        help="The checkpoint save strategy to use.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,  # NEW: Save every 500 steps
        help="Number of updates steps before saving checkpoint.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,  # NEW: Keep only best 3 checkpoints
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,  # NEW: Load best checkpoint at end
        help="Whether to load the best model at the end of training.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_map",  # NEW: Use mAP as best metric
        help="Metric to use for selecting the best model.",
    )
    
    # ========================================================================
    # Logging Configuration
    # ========================================================================
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,  # NEW: Log every 50 steps
        help="Number of update steps between logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",  # NEW: Report to tensorboard
        help="Where to report results (tensorboard, wandb, etc.).",
    )
    
    # ========================================================================
    # Output and Hub Configuration
    # ========================================================================
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,  # Changed from None to 42 for reproducibility
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_strategy",
        type=str,
        default="every_save",  # NEW: Push on every save
        choices=["end", "every_save", "checkpoint", "all_checkpoints"],
        help="Strategy to upload checkpoints to hub.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    
    # ========================================================================
    # Model-Specific Configuration
    # ========================================================================
    parser.add_argument(
        "--num_labels",
        type=int,
        default=12,  # Your 12 classes
        help="Number of labels for the pre-processor and the model.",
    )
    
    # ========================================================================
    # NEW: Model Architecture Configuration
    # ========================================================================
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,  # Keep DINOv3 frozen
        help="Whether to freeze the backbone (DINOv3).",
    )
    parser.add_argument(
        "--freeze_adapter",
        action="store_true",
        default=False,  # Train adapter with differential LR
        help="Whether to freeze the adapter layers.",
    )
    parser.add_argument(
        "--freeze_pixel_level",
        action="store_true",
        default=True,  # Keep pixel-level module frozen
        help="Whether to freeze the pixel-level module.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,  # Train encoder with differential LR
        help="Whether to freeze the transformer encoder.",
    )
    parser.add_argument(
        "--use_improved_adapter",
        action="store_true",
        default=True,  # Use enhanced adapter
        help="Whether to use the improved adapter with SE blocks and fusion.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=False,  # Not needed for M3 Max with 512x512
        help="Whether to use gradient checkpointing for memory savings.",
    )
    
    # ========================================================================
    # Parse and Process Arguments
    # ========================================================================
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    
    # Load JSON config if provided and merge with command line args
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        # Set defaults from config file (command line args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Calculate warmup steps if not provided
    if args.num_warmup_steps is None and hasattr(args, 'warmup_ratio'):
        # Will be calculated later when we know total steps
        args.num_warmup_steps = None
    
    # Validate required arguments
    if not args.model:
        raise ValueError("--model parameter is required (either via command line or config file)")
    if not args.dataset_name:
        raise ValueError("--dataset_name parameter is required (either via command line or config file)")
    if not args.output_dir:
        raise ValueError("--output_dir parameter is required (either via command line or config file)")
    
    # Create output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration summary
    print("\n" + "="*70)
    print("Training Configuration Summary")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {args.output_dir}")
    print(f"\nImage size: {args.image_height}x{args.image_width}")
    print(f"Batch size: {args.per_device_train_batch_size} (effective: {args.per_device_train_batch_size * args.gradient_accumulation_steps})")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Base LR: {args.learning_rate}")
    
    if args.use_differential_lr:
        print(f"\nDifferential Learning Rates:")
        print(f"  Adapter:  {args.learning_rate * args.adapter_lr_multiplier:.2e} ({args.adapter_lr_multiplier}x)")
        print(f"  Heads:    {args.learning_rate * args.head_lr_multiplier:.2e} ({args.head_lr_multiplier}x)")
        print(f"  Encoder:  {args.learning_rate * args.encoder_lr_multiplier:.2e} ({args.encoder_lr_multiplier}x)")
        print(f"  Decoder:  {args.learning_rate * args.decoder_lr_multiplier:.2e} ({args.decoder_lr_multiplier}x)")
    
    print(f"\nScheduler: {args.lr_scheduler_type}")
    print(f"Warmup ratio: {args.warmup_ratio}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Seed: {args.seed}")
    print("="*70 + "\n")

    return args


def main():
    args = parse_args()


    # Load model creation function and mask2former model name from config
    create_mask2former_dinov3_model, image_processor_model = load_model_from_config(args.model)
    logger.info(f"Using image processor: {image_processor_model}")

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_instance_segmentation_no_trainer", args)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    setup_logging(accelerator)

    # If passed along, set the training seed now.
    # We set device_specific to True as we want different data augmentation per device.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Create repository if push ot hub is specified
    repo_id = handle_repository_creation(accelerator, args)

    if args.push_to_hub:
        api = HfApi()


    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits - MODIFIED FOR MAPILLARY DATASET
    # ------------------------------------------------------------------------------------------------
    

    #TODO: Edit the dataset loading here
    # Check if dataset_name is a local directory (COCO dataset)
    if Path(args.dataset_name).is_dir():
        logger.info(f"Loading local COCO dataset from {args.dataset_name}")
        

        # 
        # Initialize image processor using the model's mask2former_model_name
        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_model,
            do_resize=True,
            size={"height": args.image_height, "width": args.image_width},
            do_reduce_labels=args.do_reduce_labels,
            reduce_labels=args.do_reduce_labels,
            token=args.hub_token,
            num_labels = args.num_labels,
            use_fast=True,
            ignore_index=args.num_labels,
        )

        rprint(f"image_processor: {image_processor}")
        
        # Define augmentations
        train_transform = A.Compose([A.NoOp()])
        
        val_transform = A.Compose([A.NoOp()])

        # Create datasets
        train_dataset = MapillaryInstanceDataset(
            args.dataset_name,
            image_processor,
            version='v2.0',
            split='training',
            transforms=train_transform,
        )

        
        # Try to load validation set, if not available use subset of training
        try:
            val_dataset = MapillaryInstanceDataset(
                args.dataset_name,
                image_processor,
                version='v2.0',
                split="validation",
                transforms=val_transform,
            )
        except FileNotFoundError:
            logger.warning("Validation dataset not found. Using 10% of training set.")
            from torch.utils.data import random_split
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        
        # Setup label mappings
        if hasattr(train_dataset, 'categories'):
            label2id = {name: cat_id for cat_id, name in train_dataset.categories.items()}
        else:
            # If using random_split, access the underlying dataset
            label2id = {name: cat_id for cat_id, name in train_dataset.dataset.categories.items()}
        
        if args.do_reduce_labels:
            label2id = {name: idx - 1 for name, idx in label2id.items() if idx != 0}
        
        id2label = {v: k for k, v in label2id.items()}
        

        # model = Mask2Former_Dinov3(
        #     label2id=label2id,
        #     id2label=id2label,
        #     freeze_backbone=True,
        #     hub_token=args.hub_token)

        # Initialize enhanced model
        model = Mask2Former_Dinov3(
            label2id=label2id,
            id2label=id2label,
            dinov3_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            expected_channels=[96, 192, 384, 768],
            freeze_backbone=True,
            use_improved_adapter=True,  # KEY: Use enhanced adapter
            freeze_adapter=False,  # KEY: Train the adapter for quick adaptation
            use_gradient_checkpointing=False  # Enable if running out of memory
        )

        print("Loaded label2id count:", len(model.label2id))

        # rprint(f"linear predictor weights check: {model.inner_model.class_predictor.weight[0][:5]}")

    else:
        # Original HuggingFace dataset loading code
        logger.info(f"Loading dataset from HuggingFace Hub: {args.dataset_name}")
        
        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code)
        
        label2id = dataset["train"][0]["semantic_class_to_id"]
        if args.do_reduce_labels:
            label2id = {name: idx for name, idx in label2id.items() if idx != 0}
            label2id = {name: idx - 1 for name, idx in label2id.items()}
        
        id2label = {v: k for k, v in label2id.items()}
        # Initialize enhanced model
        model = Mask2Former_Dinov3(
            label2id=label2id,
            id2label=id2label,
            dinov3_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            expected_channels=[96, 192, 384, 768],
            freeze_backbone=True,
            use_improved_adapter=True,  # KEY: Use enhanced adapter
            freeze_adapter=False,  # KEY: Train the adapter for quick adaptation
            use_gradient_checkpointing=False  # Enable if running out of memory
        )
        print("Loaded label2id count:", len(model.label2id))
        
        # Use image processor from model's mask2former_model_name
        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_model,
            do_resize=True,
            size={"height": args.image_height, "width": args.image_width},
            do_reduce_labels=args.do_reduce_labels,
            reduce_labels=args.do_reduce_labels,
            token=args.hub_token,
            num_labels = args.num_labels,
        )

        rprint(f"image_processor: {image_processor}")
        
        # Define image augmentations
        train_augment_and_transform = A.Compose([A.NoOp()])
        validation_transform = A.Compose([A.NoOp()])
        
        # Transform functions for batch
        train_transform_batch = partial(
            augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
        )
        validation_transform_batch = partial(
            augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
        )
        
        with accelerator.main_process_first():
            dataset["train"] = dataset["train"].with_transform(train_transform_batch)
            dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
        
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

    # DINOv3 backbone is already integrated in the model - no additional setup needed!
    logger.info("DINOv3-Mask2Former model ready to use.")

    # ------------------------------------------------------------------------------------------------
    # Create dataloaders
    # ------------------------------------------------------------------------------------------------
    
    dataloader_common_args = {
        "num_workers": args.dataloader_num_workers,
        "persistent_workers": args.dataloader_num_workers > 0,
        "pin_memory": False,
        "collate_fn": collate_fn,
    }
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.per_device_train_batch_size, 
        **dataloader_common_args
    )
    
    valid_dataloader = DataLoader(
        val_dataset, 
        shuffle=True, 
        batch_size=args.per_device_eval_batch_size, 
        **dataloader_common_args
    )

    # ------------------------------------------------------------------------------------------------
    # Define optimizer, scheduler and prepare everything with the accelerator
    # ------------------------------------------------------------------------------------------------

    optimizer = create_optimizer(model, args)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # ========================================================================
    # SCHEDULER: Use Cosine
    # ========================================================================
    from transformers import get_cosine_schedule_with_warmup

    # Calculate warmup steps
    num_warmup_steps = int(args.warmup_ratio * args.max_train_steps) if hasattr(args, 'warmup_ratio') else int(0.05 * args.max_train_steps)

    # OPTION 1: Cosine scheduler (RECOMMENDED for fine-tuning)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=0.5,  # Single cosine curve
    )
    # from transformers import get_polynomial_decay_schedule_with_warmup
    # lr_scheduler = get_polynomial_decay_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=args.max_train_steps,
    #     lr_end=args.learning_rate * 0.01,  # End at 1% of initial LR
    #     power=0.9
    # )
    rprint(f"\n{'='*70}")
    rprint(f"[bold cyan]Learning Rate Schedule[/bold cyan]")
    rprint(f"{'='*70}")
    rprint(f"Scheduler type: {args.lr_scheduler_type}")
    rprint(f"Total training steps: {args.max_train_steps:,}")
    rprint(f"Warmup steps: {num_warmup_steps:,} ({num_warmup_steps/args.max_train_steps*100:.1f}%)")
    rprint(f"Steps per epoch: {num_update_steps_per_epoch}")
    rprint(f"{'='*70}\n")

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ------------------------------------------------------------------------------------------------
    # Adding wandb to track the losses ### 
    # ------------------------------------------------------------------------------------------------

    import wandb
    wandb_config = {
        # Learning rate configuration
        "learning_rate": args.learning_rate,
        "use_differential_lr": getattr(args, 'use_differential_lr', False),
        "adapter_lr": args.learning_rate * getattr(args, 'adapter_lr_multiplier', 1.0),
        "head_lr": args.learning_rate * getattr(args, 'head_lr_multiplier', 1.0),
        "encoder_lr": args.learning_rate * getattr(args, 'encoder_lr_multiplier', 1.0),
        "decoder_lr": args.learning_rate * getattr(args, 'decoder_lr_multiplier', 1.0),
        "weight_decay": getattr(args, 'weight_decay', 0.05),
        "max_grad_norm": getattr(args, 'max_grad_norm', 1.0),
        
        # Model architecture
        "architecture": "Mask2Former (DINOv3 backbone)",
        "backbone": args.model,
        "freeze_backbone": getattr(args, 'freeze_backbone', True),
        "freeze_adapter": getattr(args, 'freeze_adapter', False),
        "freeze_encoder": getattr(args, 'freeze_encoder', False),
        "use_improved_adapter": getattr(args, 'use_improved_adapter', True),
        
        # Data configuration
        "image_height": args.image_height,
        "image_width": args.image_width,
        "dataset": args.dataset_name,
        "num_labels": getattr(args, 'num_labels', 12),
        "do_reduce_labels": args.do_reduce_labels,
        
        # Training configuration
        "epochs": args.num_train_epochs,
        "batch_size_per_device": args.per_device_train_batch_size,
        "eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "num_train_steps": args.max_train_steps,
        
        # Optimizer configuration
        "optimizer": type(optimizer).__name__,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        
        # Scheduler configuration
        "lr_scheduler_type": str(args.lr_scheduler_type),
        "num_warmup_steps": num_warmup_steps,
        "warmup_ratio": num_warmup_steps / args.max_train_steps,
        
        # System configuration
        "num_workers": args.dataloader_num_workers,
        "seed": args.seed,
        "device": str(accelerator.device),
        "num_processes": accelerator.num_processes,
        
        # Checkpointing
        "output_dir": args.output_dir,
        "checkpointing_steps": args.checkpointing_steps,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "save_total_limit": getattr(args, 'save_total_limit', 3),
        
        # Hub configuration
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "cache_dir": args.cache_dir,
    }

    rprint(f"\n{'='*70}")
    rprint(f"[bold cyan]Wandb Configuration[/bold cyan]")
    rprint(f"{'='*70}")
    for key, value in list(wandb_config.items())[:10]:  # Show first 10
        rprint(f"{key}: {value}")
    rprint(f"... and {len(wandb_config)-10} more")
    rprint(f"{'='*70}\n")

    rprint(f"args.lr_scheduler_type: {args.lr_scheduler_type}")
    model_name = os.path.basename(args.model).replace('.py', '')
    run_name = (
        f"{model_name}-"
        f"E{args.num_train_epochs}-"
        f"BS{args.per_device_train_batch_size}x{args.gradient_accumulation_steps}-"
        f"LR{args.learning_rate:.0e}-"
        f"{str(args.lr_scheduler_type).replace('SchedulerType.', '')}"
    )
    # Add differential LR tag if used
    if getattr(args, 'use_differential_lr', False):
        run_name += "-DiffLR"

    rprint(f"[cyan]Wandb run name: {run_name}[/cyan]\n")

    # run = wandb.init(
    #     entity="albarham-chalmers",
    #     project="Instance-segmentation-project",
    #     name=run_name,
    #     config=wandb_config,
    #     tags = [
    #         "instance-segmentation",
    #         "Mask2Former",
    #         "DINOv3",
    #         f"{args.num_labels}-classes",
    #         f"{args.image_height}x{args.image_width}",
    #         str(accelerator.device).split(':')[0].upper(),  # MPS, CUDA, or CPU
    #     ],
    #     # save_code=True,
    #     # id = "k7deh2da",
    #     # resume='auto',
    #     # allow_val_change=True
    #             )

    tags = [
        "instance-segmentation",
        "Mask2Former",
        "DINOv3",
        f"{args.num_labels}-classes",
        f"{args.image_height}x{args.image_width}",
        str(accelerator.device).split(':')[0].upper(),  # MPS, CUDA, or CPU
    ]

    # Add differential LR tag
    if getattr(args, 'use_differential_lr', False):
        tags.append("differential-lr")

    # Add training stage
    if args.resume_from_checkpoint:
        tags.append("resumed")
    else:
        tags.append("from-scratch")

    run = wandb.init(
        entity="albarham-chalmers",
        project="Instance-segmentation-project",
        name=run_name,
        config=wandb_config,
        tags=tags,
        resume="allow",  # Allow resuming if run crashes
        # OPTIONAL: Uncomment if you want to resume specific run
        # id="your-run-id",
        # resume="must",
    )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/trainable_percent": 100 * trainable_params / total_params,
            "model/num_update_steps_per_epoch": num_update_steps_per_epoch,
        }, step=0)
        
        rprint(f"\n[green]✓ Wandb initialized: {run.url}[/green]\n")

    # ------------------------------------------------------------------------------------------------
    # Run training with evaluation on each epoch
    # ------------------------------------------------------------------------------------------------

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Image processor: {image_processor_model}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None
    
    # Best epoch tracking
    best_epoch = -1
    best_metric = -1.0  # mAP 기준으로 추적
    best_metrics = {}

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            rprint(f"training_difference: {training_difference}")
            resume_step_str = training_difference.replace("step_", "").removesuffix("_state")
            resume_step = int(resume_step_str) * args.gradient_accumulation_steps
            # resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):

        model.train()


        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # ================== FIX START ==================
                # Remove the 'original_sizes' argument, which the model does not expect.
                # This information is only required for evaluation purposes.
                if "original_sizes" in batch:
                    batch.pop("original_sizes")
                # ================== FIX END ====================
                
                try:
                    outputs = model(**batch)
                except Exception as e:
                    rprint(f"An error occurred: {e}")
                    continue
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Log the loss value to wandb
                wandb.log({"train_loss": loss.item()}, step=completed_steps)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            # rprint(f"linear predictor weights check: {model.inner_model.class_predictor.weight[0][:5]}")

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    if args.output_dir is not None:
                        # ================== FIX START ==================
                        logger.info(f"***** Checkpointing at step {completed_steps} *****")

                        # 1. 평가 실행을 위해 모델을 eval 모드로 전환
                        model.eval()
                        
                        # 2. 현재 스텝의 성능 메트릭 계산
                        try:
                            metrics = evaluation_loop(model, image_processor, accelerator, valid_dataloader, id2label)
                            # At end of epoch evaluation, after metrics = evaluation_loop(...)
                            wandb.log({f"steps_eval_{k}": (v.item() if isinstance(v, torch.Tensor) and v.numel()==1 else v.tolist() if isinstance(v, torch.Tensor) else v)
                                for k, v in metrics.items()}, step=completed_steps)

                            logger.info(f"Metrics at step {completed_steps}: {metrics}")
                        
                        except Exception as e:
                            rprint(f"An error occurred: {e}")
                            continue

                        # 3. 다시 train 모드로 전환하여 학습 계속
                        model.train()

                        # 4. 학습 재개용 상태(state) 저장
                        state_checkpoint_dir = os.path.join(args.output_dir, f"step_{completed_steps}_state")
                        accelerator.save_state(state_checkpoint_dir)
                        logger.info(f"Saved training state to {state_checkpoint_dir}")
                        
                        # 5. 추론용 모델(model) 및 메트릭 저장
                        model_checkpoint_dir = os.path.join(args.output_dir, f"step_{completed_steps}_model")
                        
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # rprint(f"linear predictor weights check: {model.inner_model.class_predictor.weight[0][:5]}")


                        if accelerator.is_main_process:
                            # 추론용 모델 저장
                            os.makedirs(model_checkpoint_dir, exist_ok=True)
                            unwrapped_model.save_pretrained(
                                model_checkpoint_dir,
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save
                            )
                            image_processor.save_pretrained(model_checkpoint_dir)
                            logger.info(f"Saved inference-ready model to {model_checkpoint_dir}")
                            
                            # 메트릭을 JSON 파일로 저장
                            step_metrics = {
                                f"step_{completed_steps}_{k}": v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v 
                                for k, v in metrics.items()
                            }
                            with open(os.path.join(model_checkpoint_dir, f"step_{completed_steps}_metrics.json"), "w") as f:
                                json.dump(step_metrics, f, indent=2)
                            logger.info(f"Saved metrics to {model_checkpoint_dir}")
                        # ================== FIX END ====================

                    if args.push_to_hub and epoch < args.num_train_epochs - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        if accelerator.is_main_process:
                            image_processor.save_pretrained(args.output_dir)
                            api.upload_folder(
                                repo_id=repo_id,
                                commit_message=f"Training in progress epoch {epoch}",
                                folder_path=args.output_dir,
                                repo_type="model",
                                token=args.hub_token,
                            )

            if completed_steps >= args.max_train_steps:
                break

        logger.info("***** Running evaluation *****")
        metrics = evaluation_loop(model, image_processor, accelerator, valid_dataloader, id2label)

        wandb.log({f"epoch_eval_{k}": (v.item() if isinstance(v, torch.Tensor) and v.numel()==1 else v.tolist() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()}, step=completed_steps)


        logger.info(f"epoch {epoch}: {metrics}")

        # Best epoch 확인 및 업데이트
        current_metric = 0.0
        if 'map' in metrics:
            current_metric = metrics['map'].item() if isinstance(metrics['map'], torch.Tensor) else metrics['map']
        elif 'map_50' in metrics:
            current_metric = metrics['map_50'].item() if isinstance(metrics['map_50'], torch.Tensor) else metrics['map_50']
        elif 'map_75' in metrics:
            current_metric = metrics['map_75'].item() if isinstance(metrics['map_75'], torch.Tensor) else metrics['map_75']
        
        is_best_epoch = current_metric > best_metric
        if is_best_epoch:
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = dict(metrics)
            logger.info(f"🏆 새로운 BEST EPOCH: {epoch}, mAP: {current_metric:.4f}")

        # 각 epoch마다 모델 저장 (로컬)
        if args.output_dir is not None:
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                epoch_output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            
            if accelerator.is_main_process:
                image_processor.save_pretrained(epoch_output_dir)
                
                # epoch 메트릭 저장
                epoch_metrics = {f"epoch_{epoch}_{k}": v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
                with open(os.path.join(epoch_output_dir, f"epoch_{epoch}_metrics.json"), "w") as f:
                    json.dump(epoch_metrics, f, indent=2)
                
                logger.info(f"모델과 메트릭이 {epoch_output_dir}에 저장되었습니다.")

            # Best epoch 모델 저장
            if is_best_epoch:
                best_output_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_output_dir, exist_ok=True)
                
                unwrapped_model.save_pretrained(
                    best_output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )
                
                if accelerator.is_main_process:
                    image_processor.save_pretrained(best_output_dir)
                    
                    # Best 메트릭과 epoch 정보 저장
                    best_info = {
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "metric_name": "map" if "map" in metrics else "map_50" if "map_50" in metrics else "map_75",
                        "all_metrics": {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v for k, v in best_metrics.items()}
                    }
                    
                    with open(os.path.join(best_output_dir, "best_model_info.json"), "w") as f:
                        json.dump(best_info, f, indent=2)
                    
                    logger.info(f"🏆 BEST 모델이 {best_output_dir}에 저장되었습니다!")

        # Hub에 푸시하는 경우 (선택사항)
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        # Accelerator state 저장 (체크포인트)
        if args.checkpointing_steps == "epoch":
            checkpoint_dir = f"checkpoint_epoch_{epoch}"
            if args.output_dir is not None:
                checkpoint_dir = os.path.join(args.output_dir, checkpoint_dir)
            accelerator.save_state(checkpoint_dir)

    # ------------------------------------------------------------------------------------------------
    # Run evaluation on test dataset and save the model
    # ------------------------------------------------------------------------------------------------

    logger.info("***** Running evaluation on test dataset *****")
    metrics = evaluation_loop(model, image_processor, accelerator, valid_dataloader, id2label)
    
    # wandb.log({f"final_eval_{k}": (v.item() if isinstance(v, torch.Tensor) and v.numel()==1 else v.tolist() if isinstance(v, torch.Tensor) else v)
    #         for k, v in metrics.items()})

    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            # 텐서가 단일 값(scalar)을 가지면 .item()으로, 여러 값을 가지면 .tolist()로 변환
            processed_metrics[f"test_{key}"] = value.tolist() if value.numel() > 1 else value.item()
        else:
            processed_metrics[f"test_{key}"] = value

    logger.info(f"Test metrics: {processed_metrics}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            # 최종 결과에 best epoch 정보 포함
            final_results = {
                **processed_metrics,
                "training_summary": {
                    "total_epochs": args.num_train_epochs,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "best_metric_name": "map" if "map" in best_metrics else "map_50" if "map_50" in best_metrics else "map_75",
                    "best_epoch_metrics": {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() if isinstance(v, torch.Tensor) else v for k, v in best_metrics.items()} if best_metrics else {}
                }
            }
            
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(final_results, f, indent=2)
            
            # Log all test and summary metrics to wandb dashboard
            if run is not None:  # Only if wandb is initialized
                # Log all final test metrics. To avoid nested dict for training_summary, flatten them.
                wandb.log({
                    **{f"final_test_{k}": v for k, v in processed_metrics.items()},
                    # Summarize best metric info
                    "final_best_epoch": best_epoch,
                    "final_best_metric": best_metric,
                    "final_best_metric_name": final_results["training_summary"]["best_metric_name"],
                    # Optionally, flatten best_epoch_metrics
                    **{f"final_best_epoch_{k}": v for k, v in final_results["training_summary"].get("best_epoch_metrics", {}).items()}
                }, step=args.num_train_epochs)


            # Best epoch 요약 로그
            if best_epoch >= 0:
                logger.info(f"🏆 훈련 완료! BEST EPOCH: {best_epoch}, 최고 성능: {best_metric:.4f}")
                logger.info(f"BEST 모델 위치: {os.path.join(args.output_dir, 'best_model')}")
            else:
                logger.info("훈련 완료! (Best epoch 정보 없음)")

            image_processor.save_pretrained(args.output_dir)

            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                    ignore_patterns=["epoch_*"],
                )
                

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()