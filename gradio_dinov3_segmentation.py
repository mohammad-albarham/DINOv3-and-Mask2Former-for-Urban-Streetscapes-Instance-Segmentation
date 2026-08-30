#!/usr/bin/env python
"""
DINOv3 Mask2Former Inference Script with Gradio Interface
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import glob
import requests
import gradio as gr

from transformers import (
    AutoImageProcessor,
    AutoConfig
)
from rich import traceback, pretty
traceback.install(show_locals=True)
pretty.install()

from models.mask2former_dinov3_vitsmallplus import Mask2Former_Dinov3
from rich import print as rprint
from safetensors.torch import load_file

def get_font(font_size):
    font_path = "tmp/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        os.makedirs("tmp", exist_ok=True)
        print("Downloading font...")
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        r = requests.get(url)
        with open(font_path, "wb") as f:
            f.write(r.content)
    return ImageFont.truetype(font_path, font_size)

def load_model(model_path):
    print(f"Loading model: {model_path}")
    image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    rprint(f"Image processor loaded: {type(image_processor).__name__}")
    model = Mask2Former_Dinov3.from_pretrained(model_path)
    rprint("Model loaded successfully")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        model = model.to('mps')
        print("Using MPS (Apple Silicon)")
    else:
        print("Using CPU")
    return model, image_processor

def inference_and_visualize(model, image_processor, image, threshold=0.5, CLASS_NAMES=None):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL.Image or path")
    print(f"Image size: {image.size}")
    font = get_font(36)
    inputs = image_processor(images=[image], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [(image.size[1], image.size[0])]
    results = image_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
        return_binary_maps=False,
    )
    result = results[0]
    if result["segments_info"]:
        img_array = np.array(image)
        segmentation = result["segmentation"].cpu().numpy()
        if segmentation.ndim == 3 and segmentation.shape[0] == 1:
            segmentation = segmentation.squeeze(0)
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
        ]
        overlay = img_array.copy()
        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            segment_id = segment_info["id"]
            mask = (segmentation == segment_id)
            if mask.sum() > 0:
                color = colors[class_id % len(colors)]
                colored_mask = np.zeros_like(img_array)
                colored_mask[mask] = color
                overlay[mask] = (overlay[mask] * 0.6 + colored_mask[mask] * 0.4).astype(np.uint8)
        result_image = Image.fromarray(overlay).convert("RGBA")
        draw = ImageDraw.Draw(result_image, "RGBA")
        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}") if CLASS_NAMES else f"class_{class_id}"
            confidence = segment_info['score']
            color = colors[class_id % len(colors)]
            text = f"{class_name}: {confidence:.2f}"
            mask = (segmentation == segment_info["id"])
            if np.sum(mask) == 0: continue
            y_indices, x_indices = np.nonzero(mask)
            x_center = int(np.mean(x_indices))
            y_center = int(np.mean(y_indices))
            bbox = draw.textbbox((x_center, y_center), text, font=font)
            padded_bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
            draw.rectangle(padded_bbox, fill=(0, 0, 0, 128))
            draw.text((x_center, y_center), text, fill=tuple(color)+(255,), font=font)
        return result_image.convert("RGB")
    else:
        draw = ImageDraw.Draw(image)
        font = get_font(30)
        text = "No objects detected"
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 128))
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        return image

def gradio_inference(image, threshold):
    if image is None:
        return None, "Please upload an image"
    try:
        result_image = inference_and_visualize(
            model=model,
            image_processor=image_processor,
            image=image,
            threshold=threshold,
            CLASS_NAMES=CLASS_NAMES
        )
        return result_image, f"Inference completed with threshold {threshold}"
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, f"Error: {str(e)}"

def launch_gradio():
    with gr.Blocks(title="DINOv3 Mask2Former Segmentation") as interface:
        gr.Markdown("# 🎯 DINOv3 Mask2Former Instance Segmentation")
        gr.Markdown("Upload an image to perform instance segmentation using DINOv3 + Mask2Former")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold"
                )
                submit_btn = gr.Button("Run Segmentation", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Segmentation Result")
                status_text = gr.Textbox(label="Status", interactive=False)
        submit_btn.click(
            fn=gradio_inference,
            inputs=[input_image, threshold_slider],
            outputs=[output_image, status_text]
        )
        input_image.change(
            fn=gradio_inference,
            inputs=[input_image, threshold_slider],
            outputs=[output_image, status_text]
        )
        gr.Markdown("## 📝 Class Names")
        if 'CLASS_NAMES' in globals() and CLASS_NAMES:
            class_list = "\n".join([f"**{k}**: {v}" for k, v in CLASS_NAMES.items()])
            gr.Markdown(class_list)
    return interface

model = None
image_processor = None
CLASS_NAMES = None

def main():
    global model, image_processor, CLASS_NAMES
    parser = argparse.ArgumentParser(description="DINOv3 Mask2Former Inference with Gradio")
    parser.add_argument("--model_path", "-m", required=True, help="Path to model")
    parser.add_argument("--gradio", "-g", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--share", "-s", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()
    model, image_processor = load_model(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    CLASS_NAMES = {int(k): v for k, v in config.id2label.items()}
    print(f"Loaded {len(CLASS_NAMES)} classes: {list(CLASS_NAMES.values())}")
    if args.gradio:
        print("🚀 Launching Gradio interface...")
        interface = launch_gradio()
        interface.launch(share=args.share, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
