
import gradio as gr
import os
import torch

from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from PIL import Image # Import PIL Image
from pathlib import Path # Import Path

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the directory of the current script
current_dir = Path(__file__).parent

# Define paths relative to the script's directory
class_names_path = current_dir / "class_names.txt"
model_path = current_dir / "ViT_Food101_results.pth"
examples_path = current_dir / "examples"


with open(class_names_path, "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

# Create ViT model
vit_food101, vit_transforms = create_vit_model(num_classes=101,
                                              seed=42)

# Load the model state dict and move the model to the target device
vit_food101.load_state_dict(torch.load(f=model_path, map_location=torch.device(device)))
vit_food101.to(device)

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    # Ensure the image is in RGB format (some images might be grayscale)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_transformed = vit_transforms(img).unsqueeze(0)

    # Move the image tensor to the target device
    img_transformed = img_transformed.to(device)

    # Put model into evaluation mode and turn on inference mode
    vit_food101.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vit_food101(img_transformed), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

import gradio as gr
import glob
from pathlib import Path # Import Path to work with paths


# Create a title for the
title = "Food101 Vision"
description = "ViT model extractor test"
article = "By shyn1"


# Get a list of all image paths in the examples directory
# Corrected glob.glob usage to find all .jpg files directly in the examples directory
food101_example_paths = glob.glob(str(examples_path / "*.jpg"))


# Format the example image paths into a nested list as required by Gradio
# Each inner list contains a single example image path
gradio_examples = [[image_path] for image_path in food101_example_paths] # Use the correctly populated list

# Check if there are any examples before creating the Gradio interface
if not gradio_examples:
    print(f"No example images found in {examples_path}. Please ensure the examples folder is populated.")
else:
    demo = gr.Interface(fn=predict,
                         inputs=gr.Image(type="pil"),
                         outputs=[gr.Label(num_top_classes=5, label="Predictions"), gr.Number(label="Prediction time (s)")],
                         title=title,
                         description=description,
                         article=article,
                         examples=gradio_examples) # Pass the formatted list of examples

    demo.launch(debug=False)
