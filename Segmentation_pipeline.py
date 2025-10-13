# %%
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

from rich import print as rprint

DEBUG = False  # Set to False to suppress rich printing


def rprint_debug(*args, **kwargs):
    if DEBUG:
        rprint(*args, **kwargs)


# %%
# Use a smaller DINOv3 model for faster demos
model_name = "facebook/dino-vits16"
# Load the processor and model
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode


def extract_pooled_features(images):
    """Extracts the global (class token) features for a batch of images."""
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.cpu().numpy()


# def extract_patch_features(image):
#     """Extracts the dense patch features for a single image."""
#     inputs = processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)

#     rprint_debug(f"output of Dino-v3: {outputs}")
#     # The first token is the class token, so we skip it

#     rprint_debug(f"outputs.last_hidden_state: ")
#     rprint_debug(outputs.last_hidden_state)
#     patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()

#     return patch_features


# %%
# Load some sample images from public URLs
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # Cat image
]


# %%

# load_image("http://images.cocodataset.org/val2017/000000004134.jpg")

# %%
images = [load_image(url) for url in image_urls]

image = images[0]


# %%
# Improved segmentation head: MLP + upsampling for patch->pixel mapping
class SegmentationHead(nn.Module):
    def __init__(self, input_dim=384, num_classes=10, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: [N_patches, input_dim]
        logits = self.mlp(x)
        return logits


# Forward function and visualization
def patch_segmentation_demo(image, patch_features, seg_head, patch_size=16):
    rprint_debug(f"image: {type(image)}")
    # Removed CLS token:-> [196, dim] for 224x224 image, 16x16 patches
    patch_tokens = patch_features  # [196, 384]
    seg_logits = seg_head(torch.from_numpy(patch_tokens).float())  # [196, num_classes]
    seg_pred = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # [196]

    # Reshape to patch grid (14x14 for 224 image and 16x16 patch)
    grid_size = int(np.sqrt(len(seg_pred)))
    seg_map_patches = seg_pred.reshape((grid_size, grid_size))  # 14x14

    # Upsample to image size for visualization
    seg_map_upsampled = (
        F.interpolate(
            torch.from_numpy(seg_map_patches)[
                None, None, :, :
            ].float(),  # [1, 1, patch_h, patch_w]
            size=image.shape[:2],
            mode="nearest",  # mode='nearest': Performs nearest-neighbor interpolation,
            # meaning each patch label is expanded to cover its patch block,
            # with NO smoothing (so every pixel within the patch takes the patch label).
        )[0, 0]
        .numpy()
        .astype(np.uint8)
    )

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(seg_map_upsampled, cmap="tab20")
    plt.title("Predicted Segmentation")
    plt.axis("off")
    plt.show()


# %%


### Try to train the network on 10 images


import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import json

dataset_root = './mapillary_dataset'

images_dir = os.path.join(dataset_root, 'training', 'images')
labels_dir = os.path.join(dataset_root, 'training', 'v2.0', 'labels')


# Load config to get the number of classes
with open('./mapillary_dataset/config_v2.0.json', 'r') as f:
    config = json.load(f)

labels_config = config['labels']
num_classes = len(labels_config)

rprint_debug(f"Number of classes: {num_classes}")

# Rest of your training code remains the same, but add bounds checking:
def get_patch_labels(label_np, patch_size=16, num_classes=num_classes):
    h, w = label_np.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    mask_tensor = torch.from_numpy(label_np)[None, None].float()
    patch_labels_tensor = F.interpolate(
        mask_tensor, size=(grid_h, grid_w), mode="nearest"
    )[0, 0].long()
    
    # Set out-of-bounds labels to ignore index
    patch_labels = patch_labels_tensor.view(-1).numpy()
    patch_labels[patch_labels >= num_classes] = 255  # ignore index
    return patch_labels


model_name = "facebook/dino-vits16"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name)
model.eval()


class SegmentationHead_mapillary(nn.Module):
    def __init__(self, input_dim=384, num_classes=65, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits


def extract_patch_features(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()
    return patch_features  # [n_patches, dim]

# Training loop for a few images (for demonstration)

# Update your segmentation head
seg_head = SegmentationHead_mapillary(input_dim=384, num_classes=num_classes)


optimizer = torch.optim.Adam(seg_head.parameters(), lr=1e-3)

# Use ignore_index in loss
criterion = nn.CrossEntropyLoss(ignore_index=255)

# image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
# label_files = sorted(glob.glob(os.path.join(label_dir, "*.png")))

# Take only the first images


image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))[:10]
label_files = sorted(glob.glob(os.path.join(labels_dir, '*.png')))[:10]


epochs = 20  # small number for demo purposes



#%%


root = "./mapillary_dataset/training"
image_dir = os.path.join(root, "images")
label_dir = os.path.join(root, "v2.0", "labels")

# Adding wandb to track the experiments

import wandb

# Automatically set configuration dictionary using your variables
wandb_config = {
    "learning_rate": optimizer.param_groups[0]['lr'],
    "architecture": "ViT-patch-head",
    "backbone": model_name,
    "seg_head_hidden_dim": seg_head.mlp[0].out_features,
    "dataset": "MapillaryVistas",
    "input_res": 224,
    "patch_size": 16,
    "epochs": epochs,
    "num_classes": num_classes,
    "optimizer": type(optimizer).__name__,
    "ignore_index": criterion.ignore_index,
    "image_dir": image_dir,
    "label_dir": label_dir,
    "dataset_root": root
}

# Create descriptive run name
run_name = f"{model_name}-patch{wandb_config['patch_size']}-{wandb_config['architecture']}-E{epochs}-LR{wandb_config['learning_rate']}"

# Start wandb run with dynamic config and name
run = wandb.init(
    entity="albarham-chalmers",
    project="segmentation-project",
    name=run_name,
    config=wandb_config,
    tags=["segmentation", "ViT", "Mapillary", "demo"],
)

for epoch in range(epochs):
    total_loss = 0
    num_batches = 0
    for img_path, lbl_path in zip(image_files, label_files):
        image = Image.open(img_path).convert("RGB").resize((224,224))
        label = Image.open(lbl_path).resize((224,224), resample=Image.NEAREST)
        label_np = np.array(label)

        patch_features = extract_patch_features(image)
        patch_labels = get_patch_labels(label_np, patch_size=wandb_config["patch_size"], num_classes=wandb_config["num_classes"])

        patch_features_tensor = torch.from_numpy(patch_features).float()
        patch_labels_tensor = torch.from_numpy(patch_labels).long()

        seg_head.train()
        optimizer.zero_grad()
        logits = seg_head(patch_features_tensor)
        loss = criterion(logits, patch_labels_tensor)
        loss.backward()
        optimizer.step()

        # Log batch loss and sample info
        run.log({
            "batch_loss": loss.item(),
            "epoch": epoch,
            "img_name": os.path.basename(img_path),
        })

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    rprint(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

    run.log({
        "avg_loss": avg_loss,
        "epoch": epoch,
    })


# %%
import matplotlib.pyplot as plt

# # Pick an image to visualize
# test_img_path = image_files[1]
# test_lbl_path = label_files[1]


seg_head.eval()

for idx, (test_img_path, test_lbl_path) in enumerate(zip(image_files, label_files)):
    test_image = Image.open(test_img_path).convert("RGB").resize((224,224))
    test_label = Image.open(test_lbl_path).resize((224,224), resample=Image.NEAREST)
    test_label_np = np.array(test_label)
    test_patch_features = extract_patch_features(test_image)
    with torch.no_grad():
        test_patch_features_tensor = torch.from_numpy(test_patch_features).float()
        seg_logits = seg_head(test_patch_features_tensor)
        seg_pred = torch.argmax(seg_logits, dim=-1).cpu().numpy()
    grid_size = int(np.sqrt(len(seg_pred)))
    seg_map_patches = seg_pred.reshape((grid_size, grid_size))
    seg_map_upsampled = F.interpolate(
        torch.from_numpy(seg_map_patches)[None, None, :, :].float(),
        size=(224,224),
        mode='nearest'
    )[0, 0].numpy().astype(np.uint8)

    # Create a single 3-panel image with matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.array(test_image)); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(seg_map_upsampled, cmap="tab20"); axes[1].set_title("Predicted"); axes[1].axis("off")
    axes[2].imshow(test_label_np, cmap="tab20"); axes[2].set_title("Ground Truth"); axes[2].axis("off")
    plt.tight_layout()
    
    # Convert matplotlib figure to NumPy RGB image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Log the single composite image to wandb
    wandb.log({
        f"Segmentation_Result_{idx}": wandb.Image(img_array, caption=f"{os.path.basename(test_img_path)} - Original | Predicted | Ground Truth"),
    })

run.finish()

    # %%
