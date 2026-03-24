#!/usr/bin/env python
"""
Simple weight loading test for YOLOX-s with variable number of classes
"""

import os
import torch
import requests
# from yolox.model import create_yolox_s, create_yolox_l
from .model import create_yolox_s, create_yolox_l


def download_weights(save_path="yolox_s.pth", model = "yolox_s"):
    """Download YOLOX-s weights if not exists"""
    if os.path.exists(save_path):
        return save_path
    
    if model == "yolox_s":
        url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    elif model == "yolox_l":
        url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"
    else:
        raise ValueError("Unsupported model type. Use 'yolox_s' or 'yolox_l'.")
    print(f"Downloading weights from {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Downloaded: {save_path}")
    return save_path

def map_pretrained_weights(state_dict):
    """Map original YOLOX weight names to standalone implementation names"""
    mapped_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Map backbone weights
        if key.startswith('backbone.backbone.'):
            # backbone.backbone.xxx -> backbone.xxx
            new_key = key.replace('backbone.backbone.', 'backbone.')
        
        elif key.startswith('backbone.') and not key.startswith('backbone.backbone.'):
            # backbone.lateral_conv0.xxx -> neck.lateral_conv0.xxx
            # backbone.C3_p4.xxx -> neck.C3_p4.xxx
            # etc.
            new_key = key.replace('backbone.', 'neck.')
        
        # Head weights stay the same (head.xxx -> head.xxx)
        
        mapped_dict[new_key] = value
    
    return mapped_dict

def load_pretrained_weights(model, weights_path, num_classes=None, remap = True):
    """
    Load pretrained weights into model, handling different number of classes
    
    Args:
        model: YOLOX model instance
        weights_path: Path to weights file
        num_classes: Number of classes in your model (if different from pretrained)
    """
    print(f"Loading weights from {weights_path}")
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    
    # Get model's state dict
    if remap:
            state_dict = map_pretrained_weights(state_dict)
    # with open("state_dict_mapped.txt", "w") as f:
    #     for k, v in state_dict.items():
    #         f.write(f"{k}: {v.shape}\n")
    
    # Filter out classification layers if num_classes is different
    if num_classes is not None and num_classes != 80 and remap:  # 80 is COCO classes
        print(f"Adapting from 80 classes to {num_classes} classes")
        
        # Remove classification prediction layers
        filtered_dict = {}
        for k, v in state_dict.items():
            if 'cls_preds' in k:
                print(f"Skipping {k} (class-specific layer)")
                continue
            filtered_dict[k] = v
        
        state_dict = filtered_dict
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded weights:")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    return model

if __name__ == "__main__":

    # for testing
    weights_path = download_weights(save_path = "yolox_l.pth", model="yolox_l")
    num_classes = 4
    model = create_yolox_l(num_classes)
    model_dict = model.state_dict
    model = load_pretrained_weights(model, weights_path, num_classes)
    print(model(torch.randn(1, 3, 640, 640)).shape)  # Test forward pass