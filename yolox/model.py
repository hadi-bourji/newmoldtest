#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Complete YOLOX model implementation
"""

import torch
import torch.nn as nn
from yolox.backbone import CSPDarknet
from yolox.neck import YOLOPAFPN
from yolox.head import YOLOXHead


class YOLOX(nn.Module):
    """
    YOLOX model. Complete architecture with backbone, neck, and head.
    """
    
    def __init__(self, backbone=None, neck=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = CSPDarknet()
        if neck is None:
            neck = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
            
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        # Backbone: extract features
        backbone_features = self.backbone(x)
        
        # Neck: feature pyramid + path aggregation
        neck_features = self.neck(backbone_features)
        
        # Head: detection predictions
        outputs = self.head(neck_features)
        
        return outputs

    def init_weights(self):
        """Initialize model weights"""
        def init_yolo(m):
            for module in m.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eps = 1e-3
                    module.momentum = 0.03
                    
        self.apply(init_yolo)
        self.head.initialize_biases(1e-2)


class YOLOXs(YOLOX):
    """YOLOX-s model with specific configuration"""
    
    def __init__(self, num_classes=80):
        # YOLOX-s specific parameters
        depth = 0.33
        width = 0.50
        
        backbone = CSPDarknet(
            dep_mul=depth,
            wid_mul=width,
            out_features=("dark3", "dark4", "dark5"),
            depthwise=False,
            act="silu"
        )
        
        neck = YOLOPAFPN(
            depth=depth,
            width=width,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu"
        )
        
        head = YOLOXHead(
            num_classes=num_classes,
            width=width,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False
        )
        
        super().__init__(backbone, neck, head)

class YOLOXl(YOLOX):

    def __init__(self, num_classes = 80):
        # YOLOX-l specific parameters
        depth = 1.0
        width = 1.0
        
        backbone = CSPDarknet(
            dep_mul=depth,
            wid_mul=width,
            out_features=("dark3", "dark4", "dark5"),
            depthwise=False,
            act="silu"
        )
        
        neck = YOLOPAFPN(
            depth=depth,
            width=width,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu"
        )
        
        head = YOLOXHead(
            num_classes=num_classes,
            width=width,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False
        )
        
        super().__init__(backbone, neck, head)

class YOLOXm(YOLOX):

    def __init__(self, num_classes = 80):
        # YOLOX-m specific parameters
        depth = 0.67
        width = 0.75

        backbone = CSPDarknet(
            dep_mul=depth,
            wid_mul=width,
            out_features=("dark3", "dark4", "dark5"),
            depthwise=False,
            act="silu"
        )
        
        neck = YOLOPAFPN(
            depth=depth,
            width=width,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu"
        )
        
        head = YOLOXHead(
            num_classes=num_classes,
            width=width,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False
        )
        
        super().__init__(backbone, neck, head)

def create_yolox_s(num_classes=80):
    """Create YOLOX-s model"""
    model = YOLOXs(num_classes=num_classes)
    model.init_weights()
    return model

def create_yolox_l(num_classes=80):
    """Create YOLOX-s model"""
    model = YOLOXl(num_classes=num_classes)
    model.init_weights()
    return model

def create_yolox_m(num_classes=80):
    """Create YOLOX-s model"""
    model = YOLOXm(num_classes=num_classes)
    model.init_weights()
    return model

def get_model_info(model, input_size=(640, 640)):
    """Get model information including parameters and FLOPs"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        output = model(dummy_input)
        
    info = {
        'total_parameters': total_params,
        'total_parameters_M': total_params / 1e6,
        'input_shape': (1, 3, input_size[0], input_size[1]),
        'output_shape': output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    print("Creating YOLOX-s model...")
    model = create_yolox_s(num_classes=80)
    
    print("Model info:")
    info = get_model_info(model)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Model created successfully!")
