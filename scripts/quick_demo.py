#!/usr/bin/env python3
"""Quick training demo (1 epoch)"""

import sys
sys.path.append('..')

import torch
from models.unet import get_unet_model
from utils.dataset import get_voc_dataloader
from utils.training import train_epoch, validate_epoch, SegmentationLoss

def quick_demo():
    print("Quick training demo...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = get_unet_model(n_channels=3, n_classes=21).to(device)
    
    # Try to load sample data
    try:
        train_loader, val_loader = get_voc_dataloader(
            root_dir='../data/sample_voc',
            batch_size=2,
            image_size=128,
            num_workers=0
        )
        
        # Quick training
        criterion = SegmentationLoss(21)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("Training for 1 epoch...")
        train_loss, train_iou, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 21
        )
        
        print("Validating...")
        val_loss, val_iou, val_acc, _ = validate_epoch(
            model, val_loader, criterion, device, 21
        )
        
        print(f"Results: Train mIoU={train_iou:.4f}, Val mIoU={val_iou:.4f}")
        print("✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"Could not run training demo: {e}")
        print("Make sure to have proper VOC dataset for full training.")

if __name__ == '__main__':
    quick_demo()
