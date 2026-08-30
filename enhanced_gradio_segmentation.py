#!/usr/bin/env python
"""
DINOv3 Mask2Former Inference Script with Enhanced Gradio Interface
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import glob
import requests
import gradio as gr
import time
import pandas as pd

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
    """Download and load font for text rendering"""
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
    """Load DINOv3 Mask2Former model"""
    print(f"Loading model: {model_path}")
    
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    rprint(f"Image processor loaded: {type(image_processor).__name__}")
    
    # Load model
    model = Mask2Former_Dinov3.from_pretrained(model_path)
    rprint("Model loaded successfully")
    
    model.eval()
    
    # Set device
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
    """Run inference and create visualization with enhanced outputs"""
    
    # Handle PIL Image input
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL Image or file path")
    
    print(f"Image size: {image.size}")
    
    # Get font
    font = get_font(36)
    
    # Preprocessing
    inputs = image_processor(images=[image], return_tensors="pt")
    
    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-processing
    target_sizes = [(image.size[1], image.size[0])]  # (height, width)
    results = image_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
        return_binary_maps=False,
    )
    
    result = results[0]
    
    # Colors for different classes
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [255, 165, 0],  # DarkOrange
        [128, 128, 128], # Gray
        [255, 192, 203], # Pink
        [0, 128, 0],    # DarkGreen
    ]
    
    # Visualize results
    if result["segments_info"]:
        print(f"Detected objects: {len(result['segments_info'])}")
        
        # Convert to numpy
        img_array = np.array(image)
        segmentation = result["segmentation"].cpu().numpy()
        
        if segmentation.ndim == 3 and segmentation.shape[0] == 1:
            segmentation = segmentation.squeeze(0)
        
        print(f"Segmentation shape: {segmentation.shape}")
        
        # Create overlay
        overlay = img_array.copy()
        
        # Apply masks
        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            segment_id = segment_info["id"]
            
            # Create mask
            mask = (segmentation == segment_id)
            
            if mask.sum() > 0:
                color = colors[class_id % len(colors)]
                colored_mask = np.zeros_like(img_array)
                colored_mask[mask] = color
                overlay[mask] = (overlay[mask] * 0.6 + colored_mask[mask] * 0.4).astype(np.uint8)
        
        # Convert to PIL and add text
        result_image = Image.fromarray(overlay).convert("RGBA")
        draw = ImageDraw.Draw(result_image, "RGBA")
        
        # Add text labels
        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}") if CLASS_NAMES else f"class_{class_id}"
            confidence = segment_info['score']
            color = colors[class_id % len(colors)]
            text = f"{class_name}: {confidence:.2f}"
            
            # Find centroid
            mask = (segmentation == segment_info["id"])
            if np.sum(mask) == 0:
                continue
                
            y_indices, x_indices = np.nonzero(mask)
            x_center = int(np.mean(x_indices))
            y_center = int(np.mean(y_indices))
            
            # Draw text with background
            bbox = draw.textbbox((x_center, y_center), text, font=font)
            padded_bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
            draw.rectangle(padded_bbox, fill=(0, 0, 0, 128))
            draw.text((x_center, y_center), text, fill=tuple(color)+(255,), font=font)
        
        # Attach segments_info for later use
        final_result = result_image.convert("RGB")
        final_result.segments_info_list = result["segments_info"]
        return final_result
    
    else:
        print("No objects detected.")
        draw = ImageDraw.Draw(image)
        font = get_font(30)
        text = "No objects detected"
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 128))
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        image.segments_info_list = []
        return image


def create_mask_only(model, image_processor, image, threshold=0.5):
    """Create mask-only visualization"""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    inputs = image_processor(images=[image], return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = [(image.size[1], image.size[0])]
    results = image_processor.post_process_instance_segmentation(
        outputs, threshold=threshold, target_sizes=target_sizes, return_binary_maps=False
    )
    
    result = results[0]
    
    if result["segments_info"]:
        segmentation = result["segmentation"].cpu().numpy()
        if segmentation.ndim == 3 and segmentation.shape[0] == 1:
            segmentation = segmentation.squeeze(0)
        
        # Create colored mask
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
            [255, 165, 0], [128, 128, 128], [255, 192, 203], [0, 128, 0],
        ]
        
        mask_image = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        
        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            segment_id = segment_info["id"]
            mask = (segmentation == segment_id)
            color = colors[class_id % len(colors)]
            mask_image[mask] = color
        
        return Image.fromarray(mask_image)
    else:
        return Image.new('RGB', image.size, (0, 0, 0))


def gradio_inference(image, threshold, mask_transparency):
    """Enhanced Gradio wrapper function with multiple outputs"""
    if image is None:
        return None, None, None, None, "Please upload an image", None
    
    try:
        start_time = time.time()
        
        # Main inference
        result_image = inference_and_visualize(
            model=model,
            image_processor=image_processor,
            image=image,
            threshold=threshold,
            CLASS_NAMES=CLASS_NAMES
        )
        
        # Create mask-only version
        mask_only = create_mask_only(model, image_processor, image, threshold)
        
        elapsed = time.time() - start_time
        
        # Get segments info
        segments_info = getattr(result_image, 'segments_info_list', [])
        
        # Count instances per class
        counts = {}
        total_instances = 0
        for seg in segments_info:
            cid = seg.get('label_id', seg.get('label', 0))
            counts[cid] = counts.get(cid, 0) + 1
            total_instances += 1
        
        # Create count table
        table_data = []
        for cid, count in counts.items():
            cname = CLASS_NAMES.get(cid, f"Class_{cid}") if CLASS_NAMES else f"Class_{cid}"
            avg_conf = sum(seg['score'] for seg in segments_info 
                          if seg.get('label_id', seg.get('label', 0)) == cid) / count
            table_data.append({
                'Class': cname, 
                'Count': count,
                'Avg Confidence': f"{avg_conf:.3f}"
            })
        
        table_df = pd.DataFrame(table_data)
        
        # Create color legend
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
            [255, 165, 0], [128, 128, 128], [255, 192, 203], [0, 128, 0],
        ]
        
        legend_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;padding:10px;background:#f8f9fa;border-radius:8px;'>"
        for cid, cname in CLASS_NAMES.items():
            color = colors[cid % len(colors)]
            legend_html += f"""
            <div style='display:flex;align-items:center;padding:4px 8px;background:white;border-radius:4px;border:1px solid #ddd;'>
                <div style='width:20px;height:20px;background:rgb{tuple(color)};border-radius:3px;margin-right:8px;border:1px solid #333;'></div>
                <span style='font-size:12px;font-weight:500;'>{cname}</span>
            </div>
            """
        legend_html += "</div>"
        
        # Statistics summary
        stats_html = f"""
        <div style='background:#e8f5e8;padding:12px;border-radius:8px;margin:8px 0;'>
            <h4 style='margin:0 0 8px 0;color:#2d5a2d;'>📊 Detection Summary</h4>
            <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;'>
                <div><strong>Total Objects:</strong> {total_instances}</div>
                <div><strong>Classes Found:</strong> {len(counts)}</div>
                <div><strong>Inference Time:</strong> {elapsed:.3f}s</div>
                <div><strong>Threshold:</strong> {threshold}</div>
            </div>
        </div>
        """
        
        return (result_image, mask_only, table_df, legend_html, 
                f"✅ Detection completed successfully!", stats_html)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return (None, None, None, None, 
                f"❌ Error: {str(e)}", None)


def launch_gradio():
    """Launch enhanced Gradio interface"""
    
    with gr.Blocks(title="DINOv3 Mask2Former Segmentation", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎯 DINOv3 Mask2Former Instance Segmentation
        ### Upload an image to perform state-of-the-art instance segmentation
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Controls")
                input_image = gr.Image(type="pil", label="Upload Image", height=300)
                
                with gr.Accordion("⚙️ Parameters", open=True):
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="Confidence Threshold", 
                        info="Higher values = fewer but more confident detections"
                    )
                    mask_transparency = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.4, step=0.1,
                        label="Mask Transparency",
                        info="Adjust overlay transparency (currently visual only)"
                    )
                
                submit_btn = gr.Button("🚀 Run Segmentation", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎨 Results")
                
                with gr.Tabs():
                    with gr.TabItem("🖼️ Segmented Image"):
                        output_image = gr.Image(type="pil", label="Segmentation Result", height=350)
                    
                    with gr.TabItem("🎭 Mask Only"):
                        mask_output = gr.Image(type="pil", label="Segmentation Masks", height=350)
                
                status_output = gr.Textbox(label="Status", interactive=False, max_lines=2)
        
        # Second row for analysis
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Detection Analysis")
                stats_output = gr.HTML(label="Statistics")
                table_output = gr.DataFrame(label="Object Counts & Confidence")
            
            with gr.Column():
                gr.Markdown("### 🎨 Color Legend")
                legend_output = gr.HTML(label="Class Colors")
        
        # Event handlers
        submit_btn.click(
            fn=gradio_inference,
            inputs=[input_image, threshold_slider, mask_transparency],
            outputs=[output_image, mask_output, table_output, legend_output, status_output, stats_output]
        )
        
        # Auto-run on image upload
        input_image.change(
            fn=gradio_inference,
            inputs=[input_image, threshold_slider, mask_transparency],
            outputs=[output_image, mask_output, table_output, legend_output, status_output, stats_output]
        )
        
        # Footer info
        gr.Markdown("""
        ---
        **Model:** DINOv3 + Mask2Former | **Task:** Instance Segmentation | **Framework:** 🤗 Transformers + PyTorch
        """)
    
    return interface


# Global variables for Gradio
model = None
image_processor = None
CLASS_NAMES = None


def main():
    global model, image_processor, CLASS_NAMES

    parser = argparse.ArgumentParser(description="Enhanced DINOv3 Mask2Former Inference with Gradio")
    parser.add_argument("--model_path", "-m", required=True, help="Path to model")
    parser.add_argument("--gradio", "-g", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--share", "-s", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Port for Gradio interface")
    
    args = parser.parse_args()
    
    # Load model and processor
    model, image_processor = load_model(args.model_path)
    
    # Load class names
    config = AutoConfig.from_pretrained(args.model_path)
    CLASS_NAMES = {int(k): v for k, v in config.id2label.items()}
    print(f"✅ Loaded {len(CLASS_NAMES)} classes: {list(CLASS_NAMES.values())}")
    
    if args.gradio:
        print("🚀 Launching enhanced Gradio interface...")
        interface = launch_gradio()
        interface.launch(
            share=args.share, 
            server_name="0.0.0.0",
            server_port=args.port,
            show_error=True
        )
    else:
        print("Use --gradio flag to launch the interface")


if __name__ == "__main__":
    main()
