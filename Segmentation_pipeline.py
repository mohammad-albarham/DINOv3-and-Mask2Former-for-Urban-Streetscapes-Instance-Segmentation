#%%
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


DEBUG = False


def rprint_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# --- GPU SETUP ---
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
rprint_debug(f"Using device: {device}")

if torch.backends.mps.is_available():
    rprint_debug("✓ MPS (Metal Performance Shaders) available")
    rprint_debug(f"✓ PyTorch version: {torch.__version__}")
    rprint_debug(f"✓ Running on Apple Silicon/AMD GPU")
    
    # Optional: Check initial memory state
    try:
        allocated = torch.mps.current_allocated_memory() / 1e9
        rprint_debug(f"✓ Initial MPS memory allocated: {allocated:.2f} GB")
    except:
        rprint_debug("✓ MPS memory tracking available after first allocation")
elif not torch.backends.mps.is_built():
    rprint_debug("✗ MPS not available: PyTorch was not built with MPS enabled")
else:
    rprint_debug("✗ MPS not available: Requires macOS 12.3+ and Apple Silicon or AMD GPU")


# --- DATASET PATHS ---
dataset_root = './mapillary_dataset'
images_dir = os.path.join(dataset_root, 'training', 'images')
labels_dir = os.path.join(dataset_root, 'training', 'v2.0', 'labels')


# --- LABEL CONFIG ---
with open('./mapillary_dataset/config_v2.0.json', 'r') as f:
    config = json.load(f)
labels_config = config['labels']
num_classes = len(labels_config)


rprint_debug(f"num of classes: {num_classes}")


def get_patch_labels(label_np, patch_size=16, num_classes=num_classes):
    h, w = label_np.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    mask_tensor = torch.from_numpy(label_np)[None, None].float()
    patch_labels_tensor = F.interpolate(
        mask_tensor, size=(grid_h, grid_w), mode="nearest"
    )[0, 0].long()
    patch_labels = patch_labels_tensor.view(-1).numpy()
    patch_labels[patch_labels >= num_classes] = 255
    return patch_labels


# --- VIT BACKBONE ---
model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"


from transformers import AutoImageProcessor, AutoModel


processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name,device_map="auto")

# Move model to GPU
# model = model.to(device)
model.eval()

rprint_debug(f"Model loaded on: {next(model.parameters()).device}")

patch_size = model.config.patch_size


def extract_patch_features(image, num_register_tokens=0):
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Return GPU tensor (don't convert to numpy yet)
    patch_features = outputs.last_hidden_state[:, 1 + num_register_tokens:, :].squeeze(0)
    
    return patch_features  # [n_patches, dim] on GPU


# --- SEGMENTATION HEAD ---
class SegmentationHead_mapillary(nn.Module):
    def __init__(self, input_dim=384, num_classes=num_classes, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        logits = self.mlp(x)
        return logits


seg_head = SegmentationHead_mapillary(input_dim=384, num_classes=num_classes)

# Move segmentation head to GPU
seg_head = seg_head.to(device)
rprint_debug(f"Segmentation head loaded on: {next(seg_head.parameters()).device}")

optimizer = torch.optim.Adam(seg_head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=255)


# --- WANDb CONFIG ---
wandb_config = {
    "learning_rate": optimizer.param_groups[0]['lr'],
    "architecture": "ViT-patch-head",
    "backbone": model_name,
    "seg_head_hidden_dim": seg_head.mlp[0].out_features,
    "dataset": "MapillaryVistas",
    "input_res": 224,
    "patch_size": 16,
    "epochs": 20,
    "num_classes": num_classes,
    "optimizer": type(optimizer).__name__,
    "ignore_index": criterion.ignore_index,
    "image_dir": images_dir,
    "label_dir": labels_dir,
    "dataset_root": dataset_root,
    "device": str(device)
}


epochs = 20
run_name = f"{model_name}-patch{wandb_config['patch_size']}-{wandb_config['architecture']}-E{wandb_config['epochs']}-LR{wandb_config['learning_rate']}"


run = wandb.init(
    entity="albarham-chalmers",
    project="segmentation-project",
    name=run_name,
    config=wandb_config,
    tags=["segmentation", "ViT", "Mapillary", "demo", "GPU"],
)


# --- DATA FILES ---
image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))[:100]
label_files = sorted(glob.glob(os.path.join(labels_dir, '*.png')))[:100]


# --- TRAINING ---
for epoch in range(epochs):
    total_loss = 0
    num_batches = 0
    for img_path, lbl_path in zip(image_files, label_files):
        image = Image.open(img_path).convert("RGB").resize((224, 224))
        label = Image.open(lbl_path).resize((224, 224), resample=Image.NEAREST)
        label_np = np.array(label)

        # Extract features (returns GPU tensor)
        patch_features = extract_patch_features(image, model.config.num_register_tokens)
        patch_labels = get_patch_labels(label_np, patch_size=wandb_config["patch_size"], 
                                       num_classes=wandb_config["num_classes"])

        # Move labels to GPU
        patch_labels_tensor = torch.from_numpy(patch_labels).long().to(device)

        rprint_debug("patch_features device:", patch_features.device)
        rprint_debug("patch_features shape:", patch_features.shape)
        rprint_debug("patch_labels device:", patch_labels_tensor.device)
        rprint_debug("patch_labels shape:", patch_labels_tensor.shape)

        seg_head.train()
        optimizer.zero_grad()
        logits = seg_head(patch_features)
        loss = criterion(logits, patch_labels_tensor)
        loss.backward()
        optimizer.step()

        run.log({
            "batch_loss": loss.item(),
            "epoch": epoch,
            "img_name": os.path.basename(img_path),
        })
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    rprint_debug(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")
    run.log({"avg_loss": avg_loss, "epoch": epoch})


# --- VISUALIZE & LOG TO WANDb (panel image) ---
seg_head.eval()
for idx, (test_img_path, test_lbl_path) in enumerate(zip(image_files, label_files)):
    test_image = Image.open(test_img_path).convert("RGB").resize((224, 224))
    test_label = Image.open(test_lbl_path).resize((224, 224), resample=Image.NEAREST)
    test_label_np = np.array(test_label)
    
    # Extract features on GPU
    test_patch_features = extract_patch_features(test_image, model.config.num_register_tokens)
    
    with torch.no_grad():
        seg_logits = seg_head(test_patch_features)
        seg_pred = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # Move to CPU only for numpy
    
    grid_size = int(np.sqrt(len(seg_pred)))
    seg_map_patches = seg_pred.reshape((grid_size, grid_size))
    seg_map_upsampled = F.interpolate(
        torch.from_numpy(seg_map_patches)[None, None, :, :].float(),
        size=(224, 224),
        mode='nearest'
    )[0, 0].numpy().astype(np.uint8)

    # Create a single 3-panel image with matplotlib
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

    # Convert figure to numpy RGB image
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]  # RGB
    plt.close(fig)
    
    wandb.log({
        f"Segmentation_Result_{idx}": wandb.Image(
            img_array, 
            caption=f"{os.path.basename(test_img_path)} - Original | Predicted | Ground Truth"
        ),
    })


run.finish()

# Clean up GPU memory
torch.cuda.empty_cache()

#%%
