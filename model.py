"""
model.py — Image Preprocessing & CNN Classification Module
-----------------------------------------------------------
Handles:
  1. Image preprocessing (resize, crop, normalize)
  2. EfficientNetV2-M model loading
  3. Single-image classification
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

#Configuration

IMAGE_SIZE  = 480
NUM_CLASSES = 10
DROPOUT     = 0.4
CHECKPOINT_PATH = "ckpt/best_model.pt"

CLASS_LABELS = [
    "Eczema",                # 0
    "Viral Infections",      # 1
    "Melanoma",              # 2
    "Atopic Dermatitis",     # 3
    "Basal Cell Ca.",        # 4
    "Melanocytic Nevi",      # 5
    "Benign Keratosis",      # 6
    "Psoriasis/LP",          # 7
    "Seborrheic Keratoses",  # 8
    "Tinea/Fungal",          # 9
]

#Image Preprocessing, mimics exactly the way the model was trained

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

#Loading the model

def load_model(device: torch.device) -> nn.Module:
    """Rebuild EfficientNetV2-M architecture and load trained weights."""
    model = models.efficientnet_v2_m(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=DROPOUT * 0.7),
        nn.Linear(512, NUM_CLASSES),
    )

    try:
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state)
        print("✓ Classification model loaded!")
    except FileNotFoundError:
        print(f"⚠ WARNING: '{CHECKPOINT_PATH}' not found. Model is untrained!")

    model.to(device)
    model.eval()
    return model

#Classification

def classify_image(model: nn.Module, pil_image: Image.Image, device: torch.device) -> dict:
    """
    Run a single image through the model.
    Returns: { "disease": str, "confidence": float, "index": int }
    """
    tensor = INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    best_index = int(probs.argmax())
    return {
        "disease":    CLASS_LABELS[best_index],
        "confidence": float(probs[best_index]),
        "index":      best_index,
    }
