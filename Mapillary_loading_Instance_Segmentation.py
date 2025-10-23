#%%


# In this file, it shows how to load the Mapillary dataset for Instance segmentation
"""
PyTorch Dataset for Mapillary Vistas Instance Segmentation
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import glob

class MapillaryInstanceDataset(Dataset):
    """
    Mapillary Vistas Dataset for Instance Segmentation
    """
    def __init__(self, root_dir, version='v2.0', split='validation', transforms=None):
        """
        Args:
            root_dir: Root directory containing the dataset
            version: Dataset version ('v1.2' or 'v2.0')
            split: 'training' or 'validation'
            transforms: Transformations to apply
        """
        self.root_dir = root_dir
        self.version = version
        self.split = split
        self.transforms = transforms
        
        # Load config file to get label information
        config_path = os.path.join(root_dir, f'config_{version}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.labels = config['labels']
        

        # Check think link to understand thing and classes:
        # https://www.basic.ai/blog-post/comprehensive-guide-of-panoptic-segmentation

        # Get only "thing" classes (classes with instances)
        self.thing_classes = [label for label in self.labels if label["instances"]]
        self.label_id_to_class_id = {}

        # Create mapping from original label_id to class_id (1-indexed for PyTorch)
        # Class 0 is reserved for background
        for class_id, label in enumerate(self.labels):
            if label["instances"]:
                # +1 because 0 is background
                self.label_id_to_class_id[class_id] = len([l for l in self.labels[:class_id+1] if l["instances"]])

        # Get all image IDs
        images_dir = os.path.join(root_dir, "validation", 'images')
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]

        self.image_ids = ['--BJs76vloEaiH-wppzWNA']

        print(f"Loaded {len(self.image_ids)} images from {split} split")
        print(f"Number of thing classes (with instances): {len(self.thing_classes)}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image ID
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.root_dir, self.split, 'images', f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')

        # Load instance image
        instance_path = os.path.join(self.root_dir, self.split, self.version, 'instances', f'{image_id}.png')
        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)
        
        # Split instance_array into labels and instance IDs
        # instance_array encoding: label_id = value / 256, instance_id = value % 256
        instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
        instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)
        
        # Get unique instances (combination of label and instance ID)
        unique_instances = np.unique(instance_array)
        
        # Filter out background (0) and process each instance
        masks_list = []
        labels_list = []
        boxes_list = []
        
        for inst_value in unique_instances:
            if inst_value == 0:  # Skip background
                continue
            
            label_id = inst_value // 256
            instance_id = inst_value % 256
            
            # Only keep instances from "thing" classes
            if not self.labels[label_id]["instances"]:
                continue
            
            # Create binary mask for this instance
            mask = (instance_array == inst_value).astype(np.uint8)
            
            # Skip if mask is too small (might be noise)
            if mask.sum() < 20:  # Minimum 20 pixels
                continue
            
            masks_list.append(mask)
            
            # Get class ID (1-indexed, 0 is background)
            class_id = self.label_id_to_class_id.get(label_id, 0)
            labels_list.append(class_id)
        
        # Handle case with no instances
        if len(masks_list) == 0:
            # Create dummy data for images without instances
            masks = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            # Convert to tensors
            masks = np.stack(masks_list, axis=0)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            
            # Get bounding boxes from masks
            boxes = masks_to_boxes(masks)
            
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            
            # Calculate area
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Convert image to tensor
        image = F.to_image(image)
        
        # Prepare target dictionary
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = idx
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        
        # Apply transformations
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def get_num_classes(self):
        """Returns number of classes including background"""
        return len(self.thing_classes) + 1  # +1 for background


#%%
"""
Data Splitting and DataLoader Creation
"""

from torchvision.transforms import v2 as T

def get_transform(train):
    """
    Get transformations for training or validation
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Add more augmentations if needed
        # transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def create_dataloaders(root_dir, version='v2.0', batch_size=2, 
                       train_split=0.7, val_split=0.5, test_split=0.5,
                       num_workers=4):
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Root directory of dataset.
        version: Dataset version.
        batch_size: Batch size for training.
        train_split: Fraction of data for training (from 'training' folder).
        val_split: Fraction of data for validation (from 'validation' folder).
        test_split: Fraction of data for testing (from 'validation' folder; test set comes from validation folder).
        num_workers: Number of workers for data loading.

    Returns:
        train_loader: Dataloader for training set (from 'training' folder).
        val_loader: Dataloader for validation set (from 'validation' folder).
        test_loader: Dataloader for test set (from 'validation' folder).
        num_classes: Number of classes (including background).
    """
    # Create training dataset and split into train/test
    train_full = MapillaryInstanceDataset(
        root_dir= root_dir,
        version=version,
        split='validation',
        transforms=get_transform(train=True)
    )
    train_total = len(train_full)
    train_size = int(train_split * train_total)
    test_train_size = train_total - train_size

    print(f"\nTraining folder samples: {train_total}")
    print(f"Training: {train_size}")
    print(f"Test (from training folder): {test_train_size}")

    train_dataset, _ = torch.utils.data.random_split(
        train_full,
        [train_size, test_train_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create validation dataset and split into val/test (both come from 'validation' folder)
    val_full = MapillaryInstanceDataset(
        root_dir=root_dir,
        version=version,
        split='validation',
        transforms=get_transform(train=False)
    )
    val_total = len(val_full)
    val_size = int(val_split / (val_split + test_split) * val_total)
    test_size = val_total - val_size

    print(f"\nValidation folder samples: {val_total}")
    print(f"Validation: {val_size}")
    print(f"Testing (from validation folder): {test_size}")

    val_dataset, test_dataset = torch.utils.data.random_split(
        val_full,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    def collate_fn(batch):
        return tuple(zip(*batch))
        
        # example: 
        # batch = [(img1, target1), (img2, target2), (img3, target3)] 
        # tuple(zip(*batch))
        # # Returns: ((img1, img2, img3), (target1, target2, target3))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    num_classes = train_full.get_num_classes()

    return train_loader, val_loader, test_loader, num_classes



#%%
"""
Usage Example: Here you can understand how to load the dataset for instance segmentaion
"""

# Set your dataset root directory
DATASET_ROOT = './mapillary_dataset'  # Change this to your dataset path
VERSION = 'v2.0'

# Create dataloaders
train_loader, val_loader, test_loader, num_classes = create_dataloaders(
    root_dir=DATASET_ROOT,
    version=VERSION,
    batch_size=1,
    train_split=1.0,
    val_split=1.0,
    test_split=1.0,
    num_workers=0  # Set to 0 for debugging, increase for faster loading
)

print(f"\nNumber of classes (including background): {num_classes}")
print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

#%%
"""
Test the dataloader - visualize a sample
"""

import matplotlib.pyplot as plt

# Get one batch
images, targets = next(iter(train_loader))

print(f"\nBatch information:")
print(f"Number of images in batch: {len(images)}")
for i, (img, target) in enumerate(zip(images, targets)):
    print(f"\nImage {i}:")
    print(f"  Shape: {img.shape}")
    print(f"  Number of instances: {len(target['labels'])}")
    print(f"  Labels: {target['labels']}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    print(f"  Masks shape: {target['masks'].shape}")

#%%
"""
Helper function to get class names
"""

def get_class_names(root_dir, version='v2.0'):
    """
    Get list of class names (thing classes only)
    Returns list where index 0 is background, index 1+ are thing classes
    """
    config_path = os.path.join(root_dir, f'config_{version}.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    labels = config['labels']
    class_names = ['__background__']
    
    for label in labels:
        if label["instances"]:
            class_names.append(label["readable"])
    
    return class_names

# Get class names
class_names = get_class_names(DATASET_ROOT, VERSION)
print(f"\nClass names ({len(class_names)} total):")

limit_classes = None
for i, name in enumerate(class_names[:limit_classes]):  # Print first 10
    print(f"  {i}: {name}")
# if len(class_names) > 10:
#     print(f"  ... and {len(class_names) - 10} more classes")

#%%
import matplotlib.pyplot as plt
import numpy as np
import random

def visualize_instance_segmentation(image, target, class_names=None):
    """
    Visualize image with instance masks overlayed.
    Args:
        image: Tensor, shape (C, H, W)
        target: Dict with 'masks', 'boxes', 'labels'
        class_names: Optional, list of class names (for legend)
    """
    # Convert image tensor to numpy (H, W, C)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_np)
    ax = plt.gca()
    
    # Overlay each instance mask
    n_instances = target["masks"].shape[0]
    for i in range(n_instances):
        mask = target["masks"][i].cpu().numpy()
        label = int(target["labels"][i])
        color = [random.random() for _ in range(3)]
        
        # Mask is boolean, overlay semi-transparent
        masked_img = np.zeros((*mask.shape, 4), dtype=np.float32)
        masked_img[..., :3] = color
        masked_img[..., 3] = 0.4 * mask  # Transparency
        
        ax.imshow(masked_img, interpolation="none")

        # Draw bounding box
        box = target["boxes"][i].cpu().numpy()
        x1, y1, x2, y2 = box.astype(int)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=0.5)
        ax.add_patch(rect)
        
        # Put label
        if class_names:
            cls_name = class_names[label]
        else:
            cls_name = str(label)
        ax.text(x1, y1, cls_name, color='white', fontsize=5, bbox=dict(facecolor=tuple(color), alpha=0.7))
    
    plt.axis('off')
    plt.show()

# Usage example:
# Get one batch (already loaded above)
images, targets = next(iter(train_loader))

# Visualize first image in batch
visualize_instance_segmentation(images[0], targets[0], class_names)

# %%
