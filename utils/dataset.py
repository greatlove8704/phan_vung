import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOCSegmentationDataset(Dataset):
    """
    PASCAL VOC 2012 Segmentation Dataset
    """
    
    def __init__(self, root_dir, image_set='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        
        # Paths
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')
        
        # Read image list
        split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, f'{image_id}.png')
        mask = Image.open(mask_path)
        
        # Convert to numpy for albumentations
        image = np.array(image)
        mask = np.array(mask)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()


class CityscapesDataset(Dataset):
    """
    Cityscapes Dataset for semantic segmentation
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root_dir, 'gtFine', split)
        
        self.images = []
        self.targets = []
        
        # Collect all image and target paths
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    
                    # Find corresponding target
                    target_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    self.targets.append(os.path.join(target_dir, target_name))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx])
        
        # Convert to numpy
        image = np.array(image)
        target = np.array(target)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented['image']
            target = augmented['mask']
        
        return image, target.long()


def get_transforms(image_size=256, is_training=True):
    """
    Get augmentation transforms for training and validation
    """
    if is_training:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform


def get_voc_dataloader(root_dir, batch_size=8, image_size=256, num_workers=4):
    """
    Get PASCAL VOC dataloaders for training and validation
    """
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    train_dataset = VOCSegmentationDataset(
        root_dir=root_dir,
        image_set='train',
        transform=train_transform
    )
    
    val_dataset = VOCSegmentationDataset(
        root_dir=root_dir,
        image_set='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_cityscapes_dataloader(root_dir, batch_size=8, image_size=256, num_workers=4):
    """
    Get Cityscapes dataloaders for training and validation
    """
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    train_dataset = CityscapesDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = CityscapesDataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# VOC class names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Cityscapes class names (simplified)
CITYSCAPES_CLASSES = [
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
    'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building',
    'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle',
    'bicycle'
] 