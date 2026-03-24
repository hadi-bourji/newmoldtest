#!/usr/bin/env python
"""
Simple weight loading test for YOLOX-s with variable number of classes
"""

import os
import torch
import requests
# from yolox.model import create_yolox_s, create_yolox_l
from .model import create_yolox_s, create_yolox_l


def download_weights(save_path="./", model = "yolox_s"):
    """
    Download a YOLOX checkpoint if it does not already exist locally.

    If ``save_path`` is a directory, the file will be saved as
    ``<save_path>/<model>.pth``. If a file already exists at the target path,
    it is returned without downloading.

    :param save_path: Directory to save the weights into (a ``.pth`` filename will be appended).
    :type save_path: str
    :param model: Which model to download: ``"yolox_s"``, ``"yolox_m"``, or ``"yolox_l"``.
    :type model: str
    :return: Absolute path to the downloaded (or existing) weights file.
    :rtype: str
    :raises requests.exceptions.RequestException: If the HTTP request fails.
    :raises OSError: If the file cannot be written.
    """
    save_path =os.path.join(save_path, f"{model}.pth") 
    if os.path.exists(save_path):
        return save_path
    if model == "yolox_s":
        url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    elif model == "yolox_l":
        url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"
    elif model == "yolox_m":
        url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth"
    print(f"Downloading weights from {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()

    print("saving to: ", save_path)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Downloaded: {save_path}")
    return save_path

def map_pretrained_weights(state_dict):
    """
    Remap parameter names from official YOLOX checkpoints to this repository's layout.

    Mapping rules:
      * ``backbone.backbone.*`` → ``backbone.*``
      * Top-level ``backbone.*`` (that are not ``backbone.backbone.*``) → ``neck.*``
      * ``head.*`` keys are left unchanged

    :param state_dict: State dictionary loaded from a YOLOX checkpoint (e.g., via ``torch.load``).
    :type state_dict: dict[str, torch.Tensor]
    :return: A new state dictionary with keys remapped according to the above rules.
    :rtype: dict[str, torch.Tensor]
    """
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
    Load a YOLOX checkpoint into ``model``, optionally remapping keys and adapting class heads.
    NOTE: if you get more than 6 missing/unexpected keys, flip the remap parameter. This is
    required because the official YOLOX parameters follow different names than this repo.

    The checkpoint can be either a raw ``state_dict`` or a dictionary containing the key
    ``"model"``. If ``num_classes`` is provided and differs from 80 (COCO), class prediction
    layers named ``*cls_preds*`` are dropped so the remainder can be loaded non-strictly.

    :param model: Target YOLOX model instance to receive the weights.
    :type model: torch.nn.Module
    :param weights_path: Path to a ``.pth`` checkpoint file.
    :type weights_path: str
    :param num_classes: Number of classes in the target model. If set and not equal to 80
                        (when ``remap`` is ``True``), class-specific prediction layers are skipped.
    :type num_classes: int or None
    :param remap: Whether to apply :func:`map_pretrained_weights` before loading.
    :type remap: bool
    :return: The same ``model`` instance with weights loaded (``strict=False``).
    :rtype: torch.nn.Module
    :raises FileNotFoundError: If ``weights_path`` does not exist.
    :raises RuntimeError: If the loaded state cannot be applied due to incompatible shapes
                          (beyond the intentionally skipped heads).
    """
    print(f"Loading weights from {weights_path}")
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    
    # Get model's state dict
    if remap:
            state_dict = map_pretrained_weights(state_dict)
    
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
    weights_path = download_weights(save_path = "yolox_l", model="yolox_l")
    num_classes = 4
    model = create_yolox_l(num_classes)
    model_dict = model.state_dict
    model = load_pretrained_weights(model, weights_path, num_classes)
    print(model(torch.randn(1, 3, 640, 640)).shape)  # Test forward pass