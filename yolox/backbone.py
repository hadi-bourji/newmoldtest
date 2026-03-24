#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Standalone implementation of CSPDarknet backbone
"""

import torch
import torch.nn as nn
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck


class CSPDarknet(nn.Module):
    """CSPDarknet backbone used in YOLOX"""
    
    def __init__(self, dep_mul=1.0, wid_mul=1.0, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # For YOLOX-s: int(0.5 * 64) = 32
        base_depth = max(round(dep_mul * 3), 1)  # For YOLOX-s: max(round(0.33 * 3), 1) = 1

        # Stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # Dark2: 32 -> 64 channels, depth=1
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # Dark3: 64 -> 128 channels, depth=3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # Dark4: 128 -> 256 channels, depth=3
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # Dark5: 256 -> 512 channels, depth=1
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


# For YOLOX-s specific configuration
def create_yolox_s_backbone():
    """Create YOLOX-s backbone with specific parameters"""
    return CSPDarknet(
        dep_mul=0.33,  # depth multiplier
        wid_mul=0.50,  # width multiplier
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu"
    )
