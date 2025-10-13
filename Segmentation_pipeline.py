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


def rich_debug(*args, **kwargs):
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


def extract_patch_features(image):
    """Extracts the dense patch features for a single image."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    rich_debug(f"output of Dino-v3: {outputs}")
    # The first token is the class token, so we skip it

    rich_debug(f"outputs.last_hidden_state: ")
    rich_debug(outputs.last_hidden_state)
    patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()

    return patch_features


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
    rich_debug(f"image: {type(image)}")
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


# Usage example (assuming you have the variables):


seg_head = SegmentationHead(input_dim=384, num_classes=10)


image = image.convert("RGB")  # Ensure it's 3 channels
image_np = np.array(image)

# Get patch features for the cat image

patch_features = extract_patch_features(image)  # Should be (196, 384)

patch_segmentation_demo(image_np, patch_features, seg_head)

# %%
