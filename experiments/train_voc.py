#!/usr/bin/env python3
"""
Training script cho PASCAL VOC 2012 Semantic Segmentation
Sử dụng các mô hình: U-Net, DeepLab v3+, PSPNet

Usage:
    python train_voc.py --model unet --epochs 50 --batch_size 8 --lr 1e-4
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import warnings
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import get_unet_model
from models.deeplabv3plus import get_deeplabv3plus_model
from utils.dataset import get_voc_dataloader, VOC_CLASSES
from utils.training import train_model, plot_training_history
import segmentation_models_pytorch as smp

warnings.filterwarnings('ignore')

def get_model(model_name, num_classes, pretrained=True):
    """
    Get model by name
    """
    model_name = model_name.lower()
    
    if model_name == 'unet':
        if pretrained:
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
            )
        else:
            model = get_unet_model(n_channels=3, n_classes=num_classes, bilinear=True)
    
    elif model_name == 'deeplabv3plus':
        if pretrained:
            model = smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
            )
        else:
            model = get_deeplabv3plus_model(num_classes=num_classes, backbone='resnet50')
    
    elif model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
    
    elif model_name == 'segnet':
        # SegNet implementation from segmentation_models_pytorch doesn't exist
        # Use U-Net with different encoder
        model = smp.Unet(
            encoder_name="vgg16",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train semantic segmentation models on PASCAL VOC 2012')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet', 
                       choices=['unet', 'deeplabv3plus', 'pspnet', 'segnet'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained encoder')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data/VOCdevkit/VOC2012',
                       help='Path to VOC 2012 dataset')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='../experiments/results',
                       help='Output directory for results')
    parser.add_argument('--save_name', type=str, default=None,
                       help='Name for saved model (auto-generated if None)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"VOC dataset not found at {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate save name if not provided
    if args.save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_name = f"{args.model}_{timestamp}"
    
    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 50)
    for key, value in vars(args).items():
        print(f"{key:15}: {value}")
    print("-" * 50)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        train_loader, val_loader = get_voc_dataloader(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers
        )
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize model
    print(f"\nInitializing {args.model} model...")
    num_classes = len(VOC_CLASSES)
    model = get_model(args.model, num_classes, args.pretrained)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    save_path = os.path.join(args.output_dir, f"{args.save_name}.pth")
    
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            device=device,
            num_classes=num_classes,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            save_path=save_path
        )
        
        # Save training history
        history_path = os.path.join(args.output_dir, f"{args.save_name}_history.png")
        plot_training_history(history, save_path=history_path)
        
        # Save final results
        results = {
            'args': vars(args),
            'history': history,
            'best_miou': history['best_iou'],
            'num_parameters': num_params
        }
        
        results_path = os.path.join(args.output_dir, f"{args.save_name}_results.pt")
        torch.save(results, results_path)
        
        print(f"\nTraining completed!")
        print(f"Best validation mIoU: {history['best_iou']:.4f}")
        print(f"Model saved to: {save_path}")
        print(f"Results saved to: {results_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 