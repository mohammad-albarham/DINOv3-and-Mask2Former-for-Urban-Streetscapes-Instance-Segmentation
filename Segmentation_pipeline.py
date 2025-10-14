# %%
import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

DEBUG = False


def rprint_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# --- GPU SETUP ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rprint_debug(f"Using device: {device}")

if torch.backends.mps.is_available():
    rprint_debug("✓ MPS (Metal Performance Shaders) available")
    rprint_debug(f"✓ PyTorch version: {torch.__version__}")
    rprint_debug(f"✓ Running on Apple Silicon/AMD GPU")

    try:
        allocated = torch.mps.current_allocated_memory() / 1e9
        rprint_debug(f"✓ Initial MPS memory allocated: {allocated:.2f} GB")
    except:
        rprint_debug("✓ MPS memory tracking available after first allocation")

# --- DATASET PATHS ---
dataset_root = "./mapillary_dataset"
splits = ["training", "validation", "testing"]

image_file_lists = {}
label_file_lists = {}

# Control how many images to use from each split
# I can use None for full dataset 
MAX_TRAIN = 1000 # None if you want full dataset
MAX_VAL = 200
MAX_TEST = 100

for split in splits:
    images_dir = os.path.join(dataset_root, split, "images")
    labels_dir = os.path.join(dataset_root, split, "v2.0", "labels")
    
    # Slice the lists to control size
    image_file_lists[split] = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if split == 'training':
        image_file_lists[split] = image_file_lists[split][:MAX_TRAIN]
    elif split == 'validation':
        image_file_lists[split] = image_file_lists[split][:MAX_VAL]
    elif split == 'testing':
        image_file_lists[split] = image_file_lists[split][:MAX_TEST]
    
    # Match label files to images
    if os.path.isdir(labels_dir):
        label_file_lists[split] = sorted(glob.glob(os.path.join(labels_dir, "*.png")))[:len(image_file_lists[split])]
    else:
        label_file_lists[split] = [None] * len(image_file_lists[split])


# --- LABEL CONFIG ---
with open("./mapillary_dataset/config_v2.0.json", "r") as f:
    config = json.load(f)
labels_config = config["labels"]
num_classes = len(labels_config)

rprint_debug(f"num of classes: {num_classes}")


# --- CUSTOM DATASET CLASS ---
class MapillaryDataset(Dataset):
    """Custom Dataset for Mapillary segmentation task"""

    def __init__(
        self,
        image_files,
        label_files,
        processor,
        patch_size=16,
        num_classes=num_classes,
        image_size=224,
    ):
        """
        Args:
            image_files: List of image file paths
            label_files: List of label file paths
            processor: Image processor from transformers
            patch_size: Size of patches for segmentation
            num_classes: Number of segmentation classes
            image_size: Target image size
        """
        self.image_files = image_files
        self.label_files = label_files
        self.processor = processor
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = (
            Image.open(self.image_files[idx])
            .convert("RGB")
            .resize((self.image_size, self.image_size))
        )

        # Load and preprocess label
        label = Image.open(self.label_files[idx]).resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        label_np = np.array(label)

        # Get patch labels
        patch_labels = self.get_patch_labels(label_np)

        # Process image for model
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dim

        return {
            "pixel_values": pixel_values,
            "labels": torch.from_numpy(patch_labels).long(),
            "image_path": self.image_files[idx],
        }

    def get_patch_labels(self, label_np):
        h, w = label_np.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        mask_tensor = torch.from_numpy(label_np)[None, None].float()
        patch_labels_tensor = F.interpolate(
            mask_tensor, size=(grid_h, grid_w), mode="nearest"
        )[0, 0].long()
        patch_labels = patch_labels_tensor.view(-1).numpy()
        patch_labels[patch_labels >= self.num_classes] = 255
        return patch_labels


# --- VIT BACKBONE ---
model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"

from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name, device_map="auto")
model.eval()

rprint_debug(f"Model loaded on: {next(model.parameters()).device}")
patch_size = model.config.patch_size


def extract_patch_features(pixel_values, num_register_tokens=0):
    """Extract patch features from preprocessed pixel values"""
    # pixel_values already on correct device from dataloader
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # Return patch features (excluding CLS and register tokens)
    patch_features = outputs.last_hidden_state[:, 1 + num_register_tokens :, :]
    return patch_features  # [batch_size, n_patches, dim]


# --- SEGMENTATION HEAD ---
class SegmentationHead_mapillary(nn.Module):
    def __init__(self, input_dim=384, num_classes=num_classes, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: [batch_size, n_patches, input_dim]
        logits = self.mlp(x)  # [batch_size, n_patches, num_classes]
        return logits


seg_head = SegmentationHead_mapillary(input_dim=384, num_classes=num_classes)
seg_head = seg_head.to(device)
rprint_debug(f"Segmentation head loaded on: {next(seg_head.parameters()).device}")

optimizer = torch.optim.Adam(seg_head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=255)


# --- DATASET CREATION ---
train_dataset = MapillaryDataset(
    image_files=image_file_lists["training"],
    label_files=label_file_lists["training"],
    processor=processor,
    patch_size=patch_size,
    num_classes=num_classes,
    image_size=224,
)
val_dataset = MapillaryDataset(
    image_files=image_file_lists["validation"],
    label_files=label_file_lists["validation"],
    processor=processor,
    patch_size=patch_size,
    num_classes=num_classes,
    image_size=224,
)
test_dataset = MapillaryDataset(
    image_files=image_file_lists["testing"],
    label_files=label_file_lists["testing"],
    processor=processor,
    patch_size=patch_size,
    num_classes=num_classes,
    image_size=224,
)

print(f"Train set: {len(train_dataset)} images")
print(f"Validation set: {len(val_dataset)} images")
print(f"Test set: {len(test_dataset)} images")


# --- DATALOADERS ---

batch_size = 64  # Adjust as needed
num_workers = 0  # Set to >0 for CPU/CUDA

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
)

# --- WANDB CONFIG ---

epochs = 1


wandb_config = {
    "learning_rate": optimizer.param_groups[0]["lr"],
    "architecture": "ViT-patch-head",
    "backbone": model_name,
    "seg_head_hidden_dim": seg_head.mlp[0].out_features,
    "dataset": "MapillaryVistas",
    "input_res": 224,
    "patch_size": 16,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_classes": num_classes,
    "optimizer": type(optimizer).__name__,
    "ignore_index": criterion.ignore_index,
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "test_size": len(test_dataset),
    "dataset_root": dataset_root,
    "device": str(device),
}

run_name = f"{model_name}-patch{wandb_config['patch_size']}-{wandb_config['architecture']}-E{wandb_config['epochs']}-BS{batch_size}-LR{wandb_config['learning_rate']}"

run = wandb.init(
    entity="albarham-chalmers",
    project="segmentation-project",
    name=run_name,
    config=wandb_config,
    tags=["segmentation", "ViT", "Mapillary", "10k-images", "GPU"],
    save_code=True,  # Saves the current script
)


# --- TRAINING AND VALIDATION FUNCTIONS ---
def train_one_epoch(epoch, model, seg_head, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    seg_head.train()
    model.eval()  # Keep backbone frozen

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

    for batch in progress_bar:
        # Move batch to device
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Extract features
        patch_features = extract_patch_features(
            pixel_values, num_register_tokens=model.config.num_register_tokens
        )

        # Forward pass
        optimizer.zero_grad()
        logits = seg_head(patch_features)  # [batch_size, n_patches, num_classes]

        # Reshape for loss calculation
        batch_size, n_patches, n_classes = logits.shape
        logits_flat = logits.view(batch_size * n_patches, n_classes)
        labels_flat = labels.view(batch_size * n_patches)

        # Compute loss
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

        # Log to wandb
        wandb.log(
            {
                "batch_loss": loss.item(),
                "epoch": epoch,
            }
        )

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(epoch, model, seg_head, val_loader, criterion, device):
    """Validate the model"""
    seg_head.eval()
    model.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Extract features
            patch_features = extract_patch_features(
                pixel_values, num_register_tokens=model.config.num_register_tokens
            )

            # Forward pass
            logits = seg_head(patch_features)

            # Reshape for loss calculation
            batch_size, n_patches, n_classes = logits.shape
            logits_flat = logits.view(batch_size * n_patches, n_classes)
            labels_flat = labels.view(batch_size * n_patches)

            # Compute loss
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


# --- TRAINING LOOP ---
best_val_loss = float("inf")

for epoch in range(epochs):
    # Train
    train_loss = train_one_epoch(
        epoch, model, seg_head, train_loader, optimizer, criterion, device
    )

    # Validate
    val_loss = validate(epoch, model, seg_head, val_loader, criterion, device)

    # Log epoch metrics
    print(
        f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )
    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": seg_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            "best_model.pth",
        )
        print(f"✓ Saved best model with val_loss: {val_loss:.4f}")

# --- TESTING ---
if len([l for l in label_file_lists['testing'] if l is not None]) > 0:
    print("\n=== Testing on Test Set ===")
    test_loss = validate(-1, model, seg_head, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})
else:
    print("\n=== Test set has no public labels, skipping evaluation ===")


# --- VISUALIZE RESULTS ---
seg_head.eval()
model.eval()

# Visualize a few test samples
num_vis_samples = min(10, len(test_dataset))
test_samples_indices = np.random.choice(
    len(test_dataset), num_vis_samples, replace=False
)

for idx, sample_idx in enumerate(test_samples_indices):
    sample = test_dataset[sample_idx]

    # Get original image and label
    image_path = sample["image_path"]
    test_image = Image.open(image_path).convert("RGB").resize((224, 224))
    label_path = image_path.replace("/images/", "/v2.0/labels/").replace(".jpg", ".png")
    test_label = Image.open(label_path).resize((224, 224), resample=Image.NEAREST)
    test_label_np = np.array(test_label)

    # Predict
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)

    with torch.no_grad():
        patch_features = extract_patch_features(
            pixel_values, num_register_tokens=model.config.num_register_tokens
        )
        logits = seg_head(patch_features)
        seg_pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    # Reshape prediction
    grid_size = int(np.sqrt(len(seg_pred)))
    seg_map_patches = seg_pred.reshape((grid_size, grid_size))
    seg_map_upsampled = (
        F.interpolate(
            torch.from_numpy(seg_map_patches)[None, None, :, :].float(),
            size=(224, 224),
            mode="nearest",
        )[0, 0]
        .numpy()
        .astype(np.uint8)
    )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.array(test_image))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(seg_map_upsampled, cmap="tab20")
    axes[1].set_title("Predicted")
    axes[1].axis("off")
    axes[2].imshow(test_label_np, cmap="tab20")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")
    plt.tight_layout()

    # Convert to numpy
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    plt.close(fig)

    wandb.log(
        {
            f"Test_Result_{idx}": wandb.Image(
                img_array,
                caption=f"{os.path.basename(image_path)} - Original | Predicted | GT",
            ),
        }
    )

run.finish()
print("\nTraining complete!")

# %%
