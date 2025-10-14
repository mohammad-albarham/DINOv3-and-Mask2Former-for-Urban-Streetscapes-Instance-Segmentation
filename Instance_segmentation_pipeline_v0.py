#%%
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
#%%
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

#%%
pretrained_model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

print(processor)

model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

#%%
inputs = processor(images=image, return_tensors="pt").to(model.device)

#%%

for itm, val in inputs.items():
    print(itm)
    print(val.size())

#%%


#%%
with torch.inference_mode():
    outputs = model(**inputs)


#%%
from rich import inspect

inspect(outputs, methods=True)

#%%

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
#%%

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

#%%
pretrained_model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

#%%


inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)

# %%


model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

if torch.backends.mps.is_available():
    current_mem = torch.mps.current_allocated_memory()
    max_mem = torch.mps.driver_allocated_memory()
    print("Current GPU memory occupied by tensors (MB):", current_mem / 1024**2)
    print("Max GPU memory allocated by Metal driver (MB):", max_mem / 1024**2)


# %%