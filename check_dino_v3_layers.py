#%%
import torch
from transformers import AutoModel

#%%
# Dummy input image (batch_size=1, channels=3, height=512, width=512)
dummy_image = torch.randn(1, 3, 512, 512)

model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
model = AutoModel.from_pretrained(model_name)

# Forward pass to get hidden states
outputs = model(pixel_values=dummy_image, output_hidden_states=True, return_dict=True)


#%%
outputs.last_hidden_state.shape
#%%
hidden_states = outputs.hidden_states

#%%
len(hidden_states)
#%%
# Print all hidden state shapes
for i, h in enumerate(hidden_states):
    print(f"Hidden state {i} shape: {h.shape}")

# For a selected layer (e.g., layer 2)
layer_idx = 11
layer_output = hidden_states[layer_idx + 1]  # Adjust offset if needed
print("Selected layer_output shape from {layer_idx}:", layer_output.shape)

# Print first few tokens to see CLS, register, patches
print("First token shape (likely CLS):", layer_output[:, 0:1, :].shape)
print("Second token shape (possibly register):", layer_output[:, 1:2, :].shape)
print("First patch token shape:", layer_output[:, -1:, :].shape)

#%%
# Compute expected patch grid
batch_size, _, height, width = dummy_image.shape
patch_size = model.config.patch_size
patch_height, patch_width = height // patch_size, width // patch_size
print("Expected patch grid size:", patch_height * patch_width)

print(f"patch_height: {patch_height}")
print(f"patch_width: {patch_width}")


# Reshape just the patch tokens to feature map
patch_tokens = layer_output[:, 5:, :]  # Change index if more than 1 special token!

print(f"patch_tokens.shape: {patch_tokens.shape}")

b, n_patches, c = patch_tokens.shape
print("Number of tokens excluding CLS:", n_patches)
assert n_patches == patch_height * patch_width, "Patch tokens do not match spatial grid!"

feature_map = patch_tokens.permute(0, 2, 1).reshape(batch_size, c, patch_height, patch_width)
print("Reshaped feature map shape:", feature_map.shape)


# %%
