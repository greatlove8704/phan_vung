#!/usr/bin/env python3
"""Quick test script for models"""

import sys
sys.path.append('..')

import torch
from models.unet import get_unet_model
from models.deeplabv3plus import get_deeplabv3plus_model

def test_models():
    print("Testing models...")
    
    # Test U-Net
    unet = get_unet_model(n_channels=3, n_classes=21)
    x = torch.randn(1, 3, 256, 256)
    out = unet(x)
    print(f"U-Net: {x.shape} -> {out.shape}")
    
    # Test DeepLab v3+
    deeplab = get_deeplabv3plus_model(num_classes=21)
    out = deeplab(x)
    print(f"DeepLab v3+: {x.shape} -> {out.shape}")
    
    print("✓ All models working correctly!")

if __name__ == '__main__':
    test_models()
