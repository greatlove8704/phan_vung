#!/usr/bin/env python3
"""
Script để setup demo nhanh cho dự án Semantic Segmentation
Downloads sample data và tạo các file cần thiết
"""

import os
import sys
import requests
import zipfile
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_directories():
    """Tạo các thư mục cần thiết"""
    dirs = [
        'data/demo_images',
        'data/sample_voc',
        'experiments/results',
        'experiments/checkpoints'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def download_sample_images():
    """Download một số ảnh mẫu cho demo"""
    sample_urls = [
        'https://images.unsplash.com/photo-1544568100-847a948585b9?w=400&q=80',
        'https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400&q=80',
        'https://images.unsplash.com/photo-1574144611937-0df059b5ef3e?w=400&q=80',
        'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80',
        'https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&q=80'
    ]
    
    print("\nDownloading sample images...")
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img.save(f'data/demo_images/sample_{i+1}.jpg')
                print(f"✓ Downloaded sample_{i+1}.jpg")
            else:
                print(f"✗ Failed to download sample_{i+1}.jpg")
        except Exception as e:
            print(f"✗ Error downloading sample_{i+1}.jpg: {e}")
            # Create a dummy image if download fails
            dummy_img = Image.new('RGB', (400, 300), color=(100, 150, 200))
            dummy_img.save(f'data/demo_images/sample_{i+1}.jpg')
            print(f"✓ Created dummy sample_{i+1}.jpg")

def create_sample_voc_structure():
    """Tạo cấu trúc VOC mẫu cho testing"""
    print("\nCreating sample VOC structure...")
    
    # Create directories
    voc_dirs = [
        'data/sample_voc/JPEGImages',
        'data/sample_voc/SegmentationClass',
        'data/sample_voc/ImageSets/Segmentation'
    ]
    
    for dir_path in voc_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create sample images and masks
    for i in range(5):
        # Create sample image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img)
        pil_img.save(f'data/sample_voc/JPEGImages/2007_{i:06d}.jpg')
        
        # Create sample mask
        mask = np.random.randint(0, 21, (256, 256), dtype=np.uint8)
        pil_mask = Image.fromarray(mask)
        pil_mask.save(f'data/sample_voc/SegmentationClass/2007_{i:06d}.png')
    
    # Create image sets
    with open('data/sample_voc/ImageSets/Segmentation/train.txt', 'w') as f:
        for i in range(3):
            f.write(f'2007_{i:06d}\n')
    
    with open('data/sample_voc/ImageSets/Segmentation/val.txt', 'w') as f:
        for i in range(3, 5):
            f.write(f'2007_{i:06d}\n')
    
    print("✓ Created sample VOC structure with 5 images")

def create_test_scripts():
    """Tạo các script test nhanh"""
    
    # Test model script
    test_model_script = '''#!/usr/bin/env python3
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
'''
    
    with open('scripts/test_models.py', 'w') as f:
        f.write(test_model_script)
    
    # Quick training script
    quick_train_script = '''#!/usr/bin/env python3
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
'''
    
    with open('scripts/quick_demo.py', 'w') as f:
        f.write(quick_train_script)
    
    print("✓ Created test scripts")

def create_download_instructions():
    """Tạo hướng dẫn download datasets"""
    
    instructions = '''# Hướng dẫn Download Datasets

## PASCAL VOC 2012

### Option 1: Manual Download
```bash
cd data/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
```

### Option 2: Using torchvision
```python
import torchvision.datasets as datasets

# Download automatically
voc_train = datasets.VOCSegmentation(
    root='./data', 
    year='2012', 
    image_set='train', 
    download=True
)
```

## Cityscapes Dataset

1. Register at: https://www.cityscapes-dataset.com/
2. Download:
   - leftImg8bit_trainvaltest.zip (11GB)
   - gtFine_trainvaltest.zip (241MB)
3. Extract to data/cityscapes/

## ADE20K Dataset

```bash
cd data/
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

## Quick Start với Sample Data

Nếu bạn chỉ muốn test nhanh:
```bash
python scripts/setup_demo.py
python scripts/test_models.py
python scripts/quick_demo.py
```
'''
    
    with open('DATASET_DOWNLOAD.md', 'w') as f:
        f.write(instructions)
    
    print("✓ Created dataset download instructions")

def main():
    print("🚀 Setting up Semantic Segmentation Demo")
    print("=" * 50)
    
    # Create directory structure
    create_directories()
    
    # Download sample images
    download_sample_images()
    
    # Create sample VOC structure
    create_sample_voc_structure()
    
    # Create test scripts
    create_test_scripts()
    
    # Create download instructions
    create_download_instructions()
    
    print("\n" + "=" * 50)
    print("✅ Demo setup completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test models: python scripts/test_models.py")
    print("3. Quick demo: python scripts/quick_demo.py")
    print("4. For full training, download VOC 2012 dataset (see DATASET_DOWNLOAD.md)")
    print("5. Run training: python experiments/train_voc.py --model unet --epochs 5")
    print("6. Open notebooks/demo_segmentation.ipynb for interactive demo")

if __name__ == '__main__':
    main() 