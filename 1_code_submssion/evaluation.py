#!/usr/bin/env python
# Minimal-changes evaluation for DINOv3 + Mask2Former (instance mAP)

import os, sys, json, glob, importlib.util, logging
from pathlib import Path
from functools import partial
from typing import Any, Mapping, List, Tuple, Dict

import numpy as np
from PIL import Image
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from tqdm import tqdm

from accelerate import Accelerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import JaccardIndex

from transformers import AutoImageProcessor
from transformers.image_processing_utils import BatchFeature

# Your model class
from models.mask2former_dinov3_vitsmallplus import Mask2Former_Dinov3

from rich import print as rprint
from rich import print_json
import os
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)

def compute_pixel_accuracy(pred_class_map, tgt_class_map, ignore_index=None):
    pred_flat = pred_class_map.view(-1)
    tgt_flat = tgt_class_map.view(-1)
    if ignore_index is not None:
        mask = tgt_flat != ignore_index
    else:
        mask = torch.ones_like(tgt_flat, dtype=torch.bool)  # All True tensor
    if mask.sum() == 0:
        return 1.0
    correct = (pred_flat[mask] == tgt_flat[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

def make_serializable(metrics):
    def convert(x):
        if isinstance(x, torch.Tensor):
            return x.tolist()
        if isinstance(x, float) or isinstance(x, int) or isinstance(x, str):
            return x
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x
    return convert(metrics)


def plot_pr_curves(pr_curves, id2label=None, out_dir="pr_curves", iou_label="IoU=0.50"):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(out_dir, exist_ok=True)

    # Define class keys to plot (skip 1 and 7, add '3+7')
    base_keys = [k for k in pr_curves.keys() 
                if isinstance(k, int) and k not in [0, 1, 3, 7,4]]  # skip background, Driveway, Bicyclist, Bicycle
    if "3+7" in pr_curves:
        plot_keys = base_keys + ["3+7"]
    else:
        plot_keys = base_keys


    cols = 2
    rows = (len(plot_keys) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axs = axs.flatten()

    special_labels = { "3+7": "Bicycle+Bicyclist" }
    for i, c in enumerate(plot_keys):
        curve = pr_curves[c]
        scores, precisions, recalls = curve['scores'], curve['precision'], curve['recall']
        label = special_labels.get(c, id2label.get(c, f"Class {c}") if id2label else f"Class {c}")
        axs[i].plot(recalls, precisions, marker=".", label=label)
        axs[i].set_title(f"{label}")
        axs[i].set_xlabel("Recall")
        axs[i].set_ylabel("Precision")
        axs[i].set_xlim([0,1])
        axs[i].set_ylim([0,1])
        axs[i].grid(True)
        axs[i].legend()
    for ax in axs[len(plot_keys):]:
        ax.axis('off')

    # plt.suptitle(f"Per-class Precision-Recall Curves ({iou_label})")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, f"pr_curves_{iou_label.replace('=','').replace('.','')}.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def load_model_from_config(model_path: str):
    """
    Dynamically load model creation function and mask2former model name from the specified Python file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, 'create_mask2former_dinov3_model'):
        raise AttributeError(f"Model file {model_path} does not contain 'create_mask2former_dinov3_model' function")

    import inspect
    func_source = inspect.getsource(module.create_mask2former_dinov3_model)
    mask2former_model_name = None
    for line in func_source.split('\n'):
        line = line.strip()
        if line.startswith('mask2former_model_name') and '=' in line:
            mask2former_model_name = line.split('=', 1)[1].strip().strip('"\'')
            break
    if not mask2former_model_name:
        logger.warning(f"Could not find mask2former_model_name in {model_path}, using default")
        mask2former_model_name = "facebook/mask2former-swin-small-coco-instance"

    logger.info(f"Successfully loaded model from: {model_path}")
    logger.info(f"  - Detected mask2former base model: {mask2former_model_name}")
    return module.create_mask2former_dinov3_model, mask2former_model_name


class MapillaryInstanceDataset(Dataset):
    """
    Mapillary Vistas Dataset for Instance Segmentation
    """
    def __init__(self, root_dir, image_processor, version='v2.0', split='training', transforms=None):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.version = version
        self.split = split
        self.transforms = transforms

        config_path = os.path.join(root_dir, f'config_{version}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.labels = config['labels']

        self.thing_classes = [label for label in self.labels if label["instances"]]
        self.label_id_to_class_id = {}

        self.categories = {}
        for class_id, label in enumerate(self.labels):
            if label["instances"]:
                self.label_id_to_class_id[class_id] = len([l for l in self.labels[:class_id+1] if l["instances"]])
                self.categories[self.label_id_to_class_id[class_id]] = label["readable"]

        images_dir = os.path.join(root_dir, split, 'images')
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]

        N_val = 300
        self.image_ids = self.image_ids[:N_val]
        self.image_files = self.image_files[:N_val]

        print(f"Loaded {len(self.image_ids)} images from {split} split")
        print(f"Number of thing classes (with instances): {len(self.thing_classes)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join(self.root_dir, self.split, 'images', f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        h, w = image.height, image.width

        instance_path = os.path.join(self.root_dir, self.split, self.version, 'instances', f'{image_id}.png')
        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)

        instance_mask = np.zeros_like(instance_array, dtype=np.uint16)
        instance_to_semantic = {}

        inst_id_counter = 1
        for inst_value in np.unique(instance_array):
            if inst_value == 0:
                continue
            label_id = inst_value // 256
            if not self.labels[label_id]["instances"]:
                continue
            instance_mask[instance_array == inst_value] = inst_id_counter
            class_id = self.label_id_to_class_id.get(label_id, 0)
            instance_to_semantic[inst_id_counter] = class_id
            inst_id_counter += 1

        orig_instance_to_semantic = instance_to_semantic.copy()

        if self.transforms is not None:
            image_np = np.array(image)
            output = self.transforms(image=image_np, mask=instance_mask)
            image = output["image"]
            instance_mask = output["mask"]

            unique_ids = np.unique(instance_mask)
            instance_to_semantic = {
                int(inst_id): orig_instance_to_semantic[int(inst_id)]
                for inst_id in unique_ids
                if inst_id > 0 and int(inst_id) in orig_instance_to_semantic
            }
        else:
            image = np.array(image)

        if 0 in instance_to_semantic:
            del instance_to_semantic[0]
        assert 0 not in instance_to_semantic, "Instance mapping incorrectly includes background!"

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
        return len(self.thing_classes) + 1  # +1 for background


def augment_and_transform_batch(
    examples: Mapping[str, Any], transform: A.Compose, image_processor: AutoImageProcessor
) -> BatchFeature:
    batch = {"pixel_values": [], "mask_labels": [], "class_labels": []}
    for pil_image, pil_annotation in zip(examples["image"], examples["annotation"]):
        image = np.array(pil_image)
        semantic_and_instance_masks = np.array(pil_annotation)[..., :2]

        output = transform(image=image, mask=semantic_and_instance_masks)
        aug_image = output["image"]
        aug_semantic_and_instance_masks = output["mask"]
        aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

        unique_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
        instance_id_to_semantic_id = {
            int(instance_id): int(semantic_id)
            for semantic_id, instance_id in unique_pairs
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
    if "original_size" in examples[0]:
        batch["original_sizes"] = [example["original_size"] for example in examples]
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

from collections import defaultdict

@torch.inference_mode()
def evaluation_loop(model, image_processor, accelerator: Accelerator, dataloader, id2label=None, num_classes: int = 12):
    """
    Instance mAP (iou_type='segm'), semantic mIoU, per-class PR curves, and mean pixel accuracy.
    Returns a dict with mAP results, mIoU, per_class_iou, PR curves, and pixel accuracy.
    """
    device = accelerator.device

    # Instance mAP metric
    metric_map = MeanAveragePrecision(iou_type="segm", class_metrics=True).to(device)

    # # Semantic mIoU metric (background=0 ignored)
    # from torchmetrics import JaccardIndex
    # jaccard = JaccardIndex(task="multiclass", num_classes=num_classes, average=None).to(device)

    # Storage to build PR curves (per-class, per-image)
    preds_by_class = defaultdict(lambda: defaultdict(list))
    gts_by_class   = defaultdict(lambda: defaultdict(list))
    pixel_accuracies = []  # <<< NEW

    model.eval()
    global_img_id = 0
    for inputs in tqdm(dataloader, total=len(dataloader), disable=not accelerator.is_local_main_process):
        original_sizes = inputs.pop("original_sizes", None)
        model_inputs = {"pixel_values": inputs["pixel_values"].to(device)}
        if "pixel_mask" in inputs:
            model_inputs["pixel_mask"] = inputs["pixel_mask"].to(device)

        try:
            outputs = model(**model_inputs)
            if original_sizes is not None:
                current_target_sizes = original_sizes
            else:
                current_target_sizes = [masks.shape[-2:] for masks in inputs["mask_labels"]]

            post_processed_output = image_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                target_sizes=current_target_sizes,
                return_binary_maps=True,
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        post_processed_predictions = []
        post_processed_targets = []

        for idx, (image_predictions, target_size) in enumerate(zip(post_processed_output, current_target_sizes)):
            th, tw = int(target_size[0]), int(target_size[1])

            # Predictions (instances for mAP and PR)
            if image_predictions["segments_info"]:
                pred_masks = image_predictions["segmentation"].to(dtype=torch.bool, device=device)  # [P,H,W]
                pred_labels = torch.tensor([x["label_id"] for x in image_predictions["segments_info"]],
                                           device=device, dtype=torch.long)
                pred_scores = torch.tensor([x["score"] for x in image_predictions["segments_info"]],
                                           device=device, dtype=torch.float32)
            else:
                pred_masks = torch.zeros([0, th, tw], dtype=torch.bool, device=device)
                pred_labels = torch.empty((0,), dtype=torch.long, device=device)
                pred_scores = torch.empty((0,), dtype=torch.float32, device=device)

            post_processed_predictions.append({"masks": pred_masks, "labels": pred_labels, "scores": pred_scores})

            # Targets (instances for mAP and PR)
            target_masks = inputs["mask_labels"][idx]
            if target_masks.ndim == 2:
                target_masks = target_masks.unsqueeze(0)
            elif target_masks.ndim == 4 and target_masks.shape[1] == 1:
                target_masks = target_masks.squeeze(1)
            elif target_masks.ndim != 3:
                raise ValueError(f"mask_labels must be [M,H,W], got {tuple(target_masks.shape)}")
            target_masks = target_masks.to(device=device, dtype=torch.float32)
            if target_masks.shape[-2:] != (th, tw):
                target_masks = F.interpolate(target_masks.unsqueeze(0), size=(th, tw), mode="bilinear")[0]
            target_masks = target_masks.to(dtype=torch.bool)
            target_labels = inputs["class_labels"][idx].to(device=device, dtype=torch.long)
            post_processed_targets.append({"masks": target_masks, "labels": target_labels})

            # Build class maps for mIoU
            tgt_class_map = torch.zeros((th, tw), dtype=torch.long, device=device)
            for m, lab in zip(target_masks, target_labels):
                if 0 <= int(lab) < num_classes:
                    tgt_class_map[m] = lab
            if pred_masks.numel() == 0:
                pred_class_map = torch.zeros((th, tw), dtype=torch.long, device=device)
            else:
                score_maps = pred_masks.float() * pred_scores.view(-1, 1, 1)
                _, argmax_idx = score_maps.max(dim=0)
                covered = pred_masks.any(dim=0)
                pred_class_map = torch.zeros((th, tw), dtype=torch.long, device=device)
                if covered.any():
                    chosen = pred_labels[argmax_idx[covered]].clamp_max(num_classes - 1)
                    pred_class_map[covered] = chosen

            # # Update mIoU per image
            # jaccard.update(pred_class_map.unsqueeze(0), tgt_class_map.unsqueeze(0))
            
            # NEW --- Pixel accuracy calculation per image
            acc = compute_pixel_accuracy(pred_class_map, tgt_class_map)
            pixel_accuracies.append(acc)

            # PR curve data storage
            img_id = global_img_id
            for c in range(num_classes):
                if c == 0: continue  # ignore background
                if c == 1: continue  # skip "Driveway"
                # MERGE Bicyclist (3) and Bicycle (7) as "3+7"
                if c == 3:
                    merged_c = "3+7"
                    sel3 = (pred_labels == 3).nonzero(as_tuple=True)[0]
                    sel7 = (pred_labels == 7).nonzero(as_tuple=True)[0]
                    for k in sel3.tolist() + sel7.tolist():
                        preds_by_class[merged_c][img_id].append((
                            float(pred_scores[k].item()),
                            pred_masks[k].detach().to("cpu")
                        ))
                    sel_gt3 = (target_labels == 3).nonzero(as_tuple=True)[0]
                    sel_gt7 = (target_labels == 7).nonzero(as_tuple=True)[0]
                    for k in sel_gt3.tolist() + sel_gt7.tolist():
                        gts_by_class[merged_c][img_id].append(
                            target_masks[k].detach().to("cpu")
                        )
                elif c == 7:
                    continue  # skip (it's merged above)
                else:
                    sel = (pred_labels == c).nonzero(as_tuple=True)[0]
                    if sel.numel():
                        for k in sel.tolist():
                            preds_by_class[c][img_id].append((
                                float(pred_scores[k].item()),
                                pred_masks[k].detach().to("cpu")
                            ))
                    sel_gt = (target_labels == c).nonzero(as_tuple=True)[0]
                    if sel_gt.numel():
                        for k in sel_gt.tolist():
                            gts_by_class[c][img_id].append(
                                target_masks[k].detach().to("cpu")
                            )

            global_img_id += 1  # next image id

        # Update instance mAP per batch
        metric_map.update(post_processed_predictions, post_processed_targets)
        torch.mps.empty_cache()
        gc.collect()


    # Final metrics
    map_results = metric_map.compute()
    # per_class_iou = jaccard.compute()
    # miou = torch.nanmean(per_class_iou)
    mean_pixel_accuracy = np.mean(pixel_accuracies)  # <<< NEW AGGREGATION

    # ---- Build PR curves per class at fixed IoU thresholds ----
    def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        return (inter / union) if union > 0 else 0.0

    def pr_curve_for_class(c: int, iou_thr: float):
        records = []
        total_gt = 0
        all_imgs = set(list(preds_by_class[c].keys()) + list(gts_by_class[c].keys()))
        for img_id in all_imgs:
            preds = preds_by_class[c].get(img_id, [])
            gts   = gts_by_class[c].get(img_id, [])
            total_gt += len(gts)
            matched = [False] * len(gts)
            preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
            for score, pmask in preds_sorted:
                best_iou, best_j = 0.0, -1
                for j, gmask in enumerate(gts):
                    if not matched[j]:
                        iou = mask_iou(pmask, gmask)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                if best_iou >= iou_thr and best_j >= 0:
                    matched[best_j] = True
                    records.append((score, 1))
                else:
                    records.append((score, 0))
        if len(records) == 0:
            return [], [], []
        records.sort(key=lambda x: x[0], reverse=True)
        scores = [r[0] for r in records]
        tps = []
        fps = []
        cum_tp = 0
        cum_fp = 0
        for _, is_tp in records:
            if is_tp:
                cum_tp += 1
            else:
                cum_fp += 1
            tps.append(cum_tp)
            fps.append(cum_fp)
        precisions = []
        recalls = []
        for tp, fp in zip(tps, fps):
            prec = tp / max(tp + fp, 1)
            rec = tp / max(total_gt, 1)
            precisions.append(prec)
            recalls.append(rec)
        return scores, precisions, recalls

    pr_curves_iou50 = {}
    pr_curves_iou75 = {}

    special_classes = list(range(num_classes))
    special_classes.remove(1)   # skip "Driveway"
    special_classes.remove(7)   # skip "Bicycle" - merged with 3
    special_classes.append("3+7")  # merged class at the end

    for c in special_classes:
        if c == 0:
            continue  # skip background
        s50, p50, r50 = pr_curve_for_class(c, iou_thr=0.50)
        s75, p75, r75 = pr_curve_for_class(c, iou_thr=0.75)
        pr_curves_iou50[c] = {"scores": s50, "precision": p50, "recall": r50}
        pr_curves_iou75[c] = {"scores": s75, "precision": p75, "recall": r75}

    # Consolidate outputs
    combined = {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in map_results.items()}
    # combined["miou"] = float(miou)
    # combined["per_class_iou"] = per_class_iou.detach().cpu().tolist()
    combined["pr_curves_iou50"] = pr_curves_iou50
    combined["pr_curves_iou75"] = pr_curves_iou75
    combined["pixel_accuracy"] = float(mean_pixel_accuracy)  # <<< NEW METRIC

    return combined



def main():
    # -------------------------------
    # Simple "args" as variables
    # -------------------------------
    model_path = "/Users/pain/Desktop/Chalmers_University_of_Technlogy/Courses/third_semster/SP5/SSY340/Project/Test_Dinov3/output_optimized_differential_lr_12_classes/dinov3-smallplus-mask2former-v1.0-3000_samples-12-classes-enhanced-diff-lr/epoch_6"         # CHANGE ME
    image_processor_model = model_path # CHANGE if different
    dataset_root = "mapillary_dataset"
    dataset_version = "v2.0"
    dataset_split = "validation"
    image_height, image_width = 512, 512
    do_reduce_labels = True
    hub_token = None
    num_labels = 12  # set to number of "thing" classes + background in your setup
    per_device_eval_batch_size = 12
    dataloader_num_workers = 12

    # -------------------------------
    # Accelerator
    # -------------------------------
    accelerator = Accelerator()

    # -------------------------------
    # Image processor
    # -------------------------------
    image_processor = AutoImageProcessor.from_pretrained(
        image_processor_model,
        do_resize=True,
        size={"height": image_height, "width": image_width},
        do_reduce_labels=do_reduce_labels,
        reduce_labels=do_reduce_labels,
        token=hub_token,
        num_labels=num_labels,
        use_fast=True,
        ignore_index=num_labels,
    )

    # -------------------------------
    # Dataset + DataLoader
    # -------------------------------
    val_transform = A.Compose([A.NoOp()])
    val_dataset = MapillaryInstanceDataset(
        dataset_root,
        image_processor,
        version=dataset_version,
        split=dataset_split,
        transforms=val_transform,
    )

    dataloader_common_args = {
        "num_workers": dataloader_num_workers,
        "persistent_workers": dataloader_num_workers > 0,
        "pin_memory": False,
        "collate_fn": collate_fn,
    }
    valid_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=per_device_eval_batch_size,
        **dataloader_common_args
    )

    # -------------------------------
    # Load trained model
    # -------------------------------
    model = Mask2Former_Dinov3.from_pretrained(model_path)

    # If using distributed/multi-GPU, you can prepare with accelerator
    model = accelerator.prepare(model)
    # Data stays in PyTorch DataLoader; tensors are moved per-batch in evaluation_loop

    from transformers import AutoConfig

    # After model initialization
    config = AutoConfig.from_pretrained(model_path)

    id2label= config.id2label

    # -------------------------------
    # Run evaluation
    # -------------------------------
    logger.info("***** Running evaluation on validation dataset *****")
    metrics = evaluation_loop(model, image_processor, accelerator, valid_dataloader, id2label=id2label, num_classes=12)

    metrics_serializable = make_serializable(metrics)
    with open("metrics_results.json", "w") as f:
        json.dump(metrics_serializable, f, indent=4)

    # with open("metrics_results.json", "r") as f:
    #     metrics = json.load(f)


    plot_pr_curves(metrics['pr_curves_iou50'], id2label, out_dir="pr_curves", iou_label="IoU=0.50")
    plot_pr_curves(metrics['pr_curves_iou75'], id2label, out_dir="pr_curves", iou_label="IoU=0.75")

    metrics.pop('pr_curves_iou50')
    metrics.pop('pr_curves_iou75')

    # Pretty print scalar metrics
    printable = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            printable[k] = float(v)
        else:
            printable[k] = v


    rprint(printable)


if __name__ == "__main__":
    main()
