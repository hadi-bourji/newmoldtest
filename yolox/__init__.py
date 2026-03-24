#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Standalone YOLOX implementation
"""

from .network_blocks import *
from .backbone import CSPDarknet, create_yolox_s_backbone
from .neck import YOLOPAFPN, create_yolox_s_neck
from .head import YOLOXHead, create_yolox_s_head
from .model import YOLOX, YOLOXs, create_yolox_s, get_model_info

__all__ = [
    # Building blocks
    'BaseConv', 'DWConv', 'Bottleneck', 'ResLayer', 'SPPBottleneck', 'CSPLayer', 'Focus',
    # Backbone
    'CSPDarknet', 'create_yolox_s_backbone',
    # Neck
    'YOLOPAFPN', 'create_yolox_s_neck', 
    # Head
    'YOLOXHead', 'create_yolox_s_head',
    # Complete models
    'YOLOX', 'YOLOXs', 'create_yolox_s', 'get_model_info'
]
