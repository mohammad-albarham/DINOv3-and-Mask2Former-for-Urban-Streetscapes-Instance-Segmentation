#!/usr/bin/env python
"""
Simple DINOv3 Mask2Former Inference Script
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os

from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    AutoConfig
)
import glob

from rich import traceback, pretty
traceback.install(show_locals=True)

pretty.install()


from models.mask2former_dinov3_vitsmallplus import Mask2Former_Dinov3



# Class ID to name mapping
# CLASS_NAMES = {
#     0: "bead 1",
#     1: "bead 2",
#     2: "bead 3",
#     3: "bead 4",
#     4: "bead 5",
#     5: "bead 6",
#     6: "bead 7",
#     7: "bead 8"
# # }
# CLASS_NAMES ={
#     0: "Bird",
#     1: "Ground Animal",
#     2: "Temporary Barrier",
#     3: "Crosswalk - Plain",
#     4: "Driveway",
#     5: "Person",
#     6: "Bicyclist",
#     7: "Motorcyclist",
#     8: "Other Rider",
#     9: "Lane Marking - Arrow (Left)",
#     10: "Lane Marking - Arrow (Other)",
#     11: "Lane Marking - Arrow (Right)",
#     12: "Lane Marking - Arrow (Split Left or Straight)",
#     13: "Lane Marking - Arrow (Split Right or Straight)",
#     14: "Lane Marking - Arrow (Straight)",
#     15: "Lane Marking - Crosswalk",
#     16: "Lane Marking - Give Way (Row)",
#     17: "Lane Marking - Give Way (Single)",
#     18: "Lane Marking - Other",
#     19: "Lane Marking - Stop Line",
#     20: "Lane Marking - Symbol (Bicycle)",
#     21: "Lane Marking - Symbol (Other)",
#     22: "Lane Marking - Text",
#     23: "Banner",
#     24: "Bench",
#     25: "Bike Rack",
#     26: "Catch Basin",
#     27: "CCTV Camera",
#     28: "Fire Hydrant",
#     29: "Junction Box",
#     30: "Mailbox",
#     31: "Manhole",
#     32: "Parking Meter",
#     33: "Phone Booth",
#     34: "Signage - Advertisement",
#     35: "Signage - Ambiguous",
#     36: "Signage - Back",
#     37: "Signage - Information",
#     38: "Signage - Other",
#     39: "Signage - Store",
#     40: "Street Light",
#     41: "Pole",
#     42: "Traffic Sign Frame",
#     43: "Utility Pole",
#     44: "Traffic Cone",
#     45: "Traffic Light - General (Single)",
#     46: "Traffic Light - Pedestrians",
#     47: "Traffic Light - General (Upright)",
#     48: "Traffic Light - General (Horizontal)",
#     49: "Traffic Light - Cyclists",
#     50: "Traffic Light - Other",
#     51: "Traffic Sign - Ambiguous",
#     52: "Traffic Sign (Back)",
#     53: "Traffic Sign - Direction (Back)",
#     54: "Traffic Sign - Direction (Front)",
#     55: "Traffic Sign (Front)",
#     56: "Traffic Sign - Parking",
#     57: "Traffic Sign - Temporary (Back)",
#     58: "Traffic Sign - Temporary (Front)",
#     59: "Trash Can",
#     60: "Bicycle",
#     61: "Boat",
#     62: "Bus",
#     63: "Car",
#     64: "Caravan",
#     65: "Motorcycle",
#     66: "On Rails",
#     67: "Other Vehicle",
#     68: "Trailer",
#     69: "Truck",
#     70: "Wheeled Slow",
#     71: "Water Valve"}

import os
import requests
from PIL import ImageFont

def get_font(font_size):
    font_path = "tmp/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        print("Downloading font...")
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"

        r = requests.get(url)
        with open(font_path, "wb") as f:
            f.write(r.content)
    return ImageFont.truetype(font_path, font_size)


def load_model(model_path):
    """Load model"""
    print(f"Loading model: {model_path}")
    
    image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    # model = AutoModelForUniversalSegmentation.from_pretrained(model_path)


    # model_instance = Mask2Former_Dinov3()

    # Mask2Former_Dinov3_config = AutoConfig.from_pretrained(model_path)

    # model = model_instance.create_mask2former_dinov3_model(
    #     label2id=Mask2Former_Dinov3_config.label2id,
    #     id2label=Mask2Former_Dinov3_config.id2label,
    #     freeze_backbone=True,
    #     hub_token=None,
    # )

    # model = Mask2Former_Dinov3.from_pretrained(model_path)
    
    from safetensors.torch import load_file

    state_dict = load_file("/Users/pain/Desktop/Chalmers_University_of_Technlogy/Courses/third_semster/SP5/SSY340/Project/Test_Dinov3/output/dinov3-smallplus-mask2former-1e4-unfreeze-1000_samples/best_model/model.safetensors")

    from rich import print as rprint
    # rprint(f"state_dict: {state_dict.inner_model.model.}")
    from transformers import AutoConfig

    # After model initialization
    config = AutoConfig.from_pretrained("/Users/pain/Desktop/Chalmers_University_of_Technlogy/Courses/third_semster/SP5/SSY340/Project/Test_Dinov3/output/dinov3-smallplus-mask2former-1e4-unfreeze-1000_samples/best_model/")

    # # Assume you have instantiated your model with the correct class mappings
    # model = Mask2Former_Dinov3(
    # label2id=config.label2id,
    # id2label=config.id2label,
    # freeze_backbone=True,
    # hub_token=config.hub_token)
    
    # model.load_state_dict(state_dict, strict=False)

    model = Mask2Former_Dinov3.from_pretrained(model_path)
    # model.load_state_dict(state_dict, strict=False)

    
    rprint(f"linear predictor weights check: {model.inner_model.class_predictor.weight[0][:5]}")


    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("using cuda gpu")
    elif torch.backends.mps.is_available():
        model = model.to('mps')
    else:
        print("Using CPU")
    
    return model, image_processor

def inference_and_visualize(model, image_processor, image_path, save_path='None', threshold=0.5, CLASS_NAMES=None):
    """Inference and visualization (modified version)"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")

    # Font setup (for text display)
    # try:
    #     # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    #     font = get_font(36)
    # except OSError:
    #     # Use default font
    #     font = ImageFont.load_default()
    font = get_font(36)

    # Preprocessing
    inputs = image_processor(images=[image], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    if torch.backends.mps.is_available():
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

    # Visualize results
    if result["segments_info"]:
        print(f"Detected objects: {len(result['segments_info'])}")

        # Convert original image to numpy array
        img_array = np.array(image)
        segmentation = result["segmentation"].cpu().numpy()

        if segmentation.ndim == 3 and segmentation.shape[0] == 1:
            segmentation = segmentation.squeeze(0)
            
        # Debug: print segmentation information
        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Segmentation unique values: {np.unique(segmentation)}")
        
        # Print each segment information
        for i, seg_info in enumerate(result["segments_info"]):
            print(f"Segment {i}: ID={seg_info['id']}, Label={seg_info.get('label_id', seg_info.get('label', 0))}, Score={seg_info['score']:.3f}")

        # Color generation (different colors for each class)
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
        ]

        # Mask overlay
        overlay = img_array.copy()
        
        # Prepare PIL Image for text overlay
        result_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(result_image)

        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            segment_id = segment_info["id"]
            
            # Create mask (original logic)
            mask = (segmentation == segment_id)
            
            # Mask debugging information
            mask_pixels = mask.sum()
            total_pixels = mask.shape[0] * mask.shape[1]
            
            print(f"Segment ID {segment_id}, Class {class_id}: Mask pixels {mask_pixels}/{total_pixels}")
            
            # Mask verification and correction
            mask_ratio = mask_pixels / total_pixels

            # --- Key modification: select color based on class_id ---
            color = colors[class_id % len(colors)]

            # Apply semi-transparent mask
            if mask.sum() > 0:
                # Create a new image filled with color in the mask area
                colored_mask = np.zeros_like(img_array)
                colored_mask[mask] = color
                
                # Blend original image with mask
                overlay[mask] = (overlay[mask] * 0.6 + colored_mask[mask] * 0.4).astype(np.uint8)

        # Final conversion to PIL Image
        result_image = Image.fromarray(overlay).convert("RGBA")
        draw = ImageDraw.Draw(result_image, "RGBA")


        y_offset = 10
        # Add text overlay
        for segment_info in result["segments_info"]:
            class_id = segment_info.get('label_id', segment_info.get('label', 0))
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            confidence = segment_info['score']
            color = colors[class_id % len(colors)]
            text = f"{class_name}: {confidence:.2f}"

            # Find centroid of mask
            mask = (segmentation == segment_info["id"])
            if np.sum(mask) == 0:
                continue
            y_indices, x_indices = np.nonzero(mask)
            x_center = int(np.mean(x_indices))
            y_center = int(np.mean(y_indices))

            bbox = draw.textbbox((x_center, y_center), text, font=font)
            padded_bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
            draw.rectangle(padded_bbox, fill=(0, 0, 0, 128))
            draw.text((x_center, y_center), text, fill=tuple(color)+(255,), font=font)


        # Save
        save_filename = save_path
        if not save_filename.endswith(".png"):
            save_filename = save_filename.rsplit('.', 1)[0] + ".png"
        result_image.save(save_filename)


        result_image.save(save_filename)
        print(f"Result saved: {save_filename}")

        return result_image

    else:
        print("No objects detected.")
        # (same as original code)
        draw = ImageDraw.Draw(image)
        try:
            font= get_font(30)
        except OSError:
            font = ImageFont.load_default()
        
        text = "No objects detected"
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 128))
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
        if save_path:
            image.save(save_path)
        return image

def process_directory(model, image_processor, input_dir, output_dir, threshold=0.5, recursive=False, CLASS_NAMES=None):
    """Batch process all images in directory"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Find all image files
    image_files = []
    
    if recursive:
        # Recursively search subdirectories
        for root, dirs, files in os.walk(input_dir):
            for ext in image_extensions:
                pattern = ext.lower()
                for file in files:
                    if file.lower().endswith(pattern[1:]):  # *.jpg -> .jpg
                        image_files.append(os.path.join(root, file))
    else:
        # Search current directory only
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    image_files.sort()
    
    print(f"Images to process: {len(image_files)} (Recursive search: {'On' if recursive else 'Off'})")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        
        # try:
        # Inference and visualization
        result_image = inference_and_visualize(
            model, image_processor, image_path, output_path, threshold, CLASS_NAMES
        )
        print(f"✅ Completed: {output_path}")
        
        # except Exception as e:
        #     print(f"❌ Error occurred: {e}")
        #     continue
    
    print(f"\n🎉 All processing completed! Results saved in {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Simple Mask2Former Inference")
    parser.add_argument("--model_path", "-m", help="Path to model", 
                        default="//Users/pain/Desktop/Chalmers_University_of_Technlogy/Courses/third_semster/SP5/SSY340/Project/Test_Dinov3/output/dinov3-smallplus-mask2former-1e4-unfreeze-1000_samples/epoch_29")
    parser.add_argument("--image_path", "-i", help="Path to single image",
                        default="/Users/pain/Desktop/Chalmers_University_of_Technlogy/Courses/third_semster/SP5/SSY340/Project/Test_Dinov3/mapillary_dataset/training/images/__IoBfs3I6vB5ND-vqXK1A.jpg")
    parser.add_argument("--input_dir", "-d", help="Input directory (for batch processing)", 
                        default="/Users/pain/Desktop/Chalmers_University_of_Technlogy/Courses/third_semster/SP5/SSY340/Project/Test_Dinov3/test_inference"
                        )
    parser.add_argument("--output_dir", "-od", help="Output directory (for batch processing)", 
                        default="/Users/pain/Desktop/Chalmers_University_of_Technlogy/Courses/third_semster/SP5/SSY340/Project/Test_Dinov3/results_temp")
    parser.add_argument("--output", "-o", help="Output path for single image result", 
                        default='output.png')
    parser.add_argument("--threshold", "-t", type=float, default=0.2, help="Detection threshold (default: 0.5)")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch processing mode")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Load model
    model, image_processor = load_model(args.model_path)
    


    from transformers import AutoConfig

    # After model initialization
    config = AutoConfig.from_pretrained(args.model_path)

    CLASS_NAMES = {int(k): v for k, v in config.id2label.items()}


    if args.batch or args.input_dir:
        # Batch processing mode
        print("🚀 Batch processing mode")
        process_directory(
            model, 
            image_processor, 
            args.input_dir, 
            args.output_dir, 
            args.threshold,
            args.recursive,
            CLASS_NAMES
        )
    else:
        # Single image processing mode
        if not args.image_path:
            print("❌ Error: --image_path is required in single image mode.")
            return
        
        print("🎯 Single image processing mode")
        result_image = inference_and_visualize(
            model, 
            image_processor, 
            args.image_path, 
            args.output, 
            args.threshold,
            CLASS_NAMES
        )
        print("✅ Completed!")

if __name__ == "__main__":
    main()
