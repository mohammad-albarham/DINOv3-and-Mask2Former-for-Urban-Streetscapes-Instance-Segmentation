#%%
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import numpy as np

#%%
# Use a smaller DINOv3 model for faster demos
model_name = "facebook/dino-vits16"
# Load the processor and model
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name)
model.eval() # Set the model to evaluation mode

#%%

import torch.nn as nn

from rich import print as rprint 


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

    rprint(f"output of Dino-v3: {outputs}")
    # The first token is the class token, so we skip it

    rprint(f"outputs.last_hidden_state: ")
    rprint(outputs.last_hidden_state)
    patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()


    return patch_features

#%%
# Load some sample images from public URLs
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # Cat image
    "http://images.cocodataset.org/val2017/000000004134.jpg" # Bird image
]


#%%

# load_image("http://images.cocodataset.org/val2017/000000004134.jpg")

#%%
images = [load_image(url) for url in image_urls]

#%%

print("\n--- 5. Semantic Segmentation (with a simple head) ---")

class SegmentationHead(nn.Module):
    def __init__(self, input_dim=384, num_classes=10):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.head(x)

# Create a dummy segmentation head
seg_head = SegmentationHead()
seg_head.eval()

# Get patch features for the cat image
patch_features = extract_patch_features(images[0])

rprint(f"Patch features are: {patch_features.shape}")

# Get segmentation map
with torch.no_grad():
    segmentation_logits = seg_head(torch.from_numpy(patch_features))

    rprint(f"segmentation_logits: {segmentation_logits.shape}")

    segmentation_map = torch.argmax(segmentation_logits, dim=-1)

print("Shape of the segmentation map:", segmentation_map.shape)
# %%