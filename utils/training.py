import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class SegmentationLoss(nn.Module):
    """
    Combined loss for semantic segmentation
    """
    
    def __init__(self, num_classes, ignore_index=255, weight=None):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        
    def forward(self, predictions, targets):
        # Cross entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        return ce_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    """
    
    def __init__(self, num_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Convert to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate dice coefficient for each class
        dice_scores = []
        for i in range(self.num_classes):
            pred_i = predictions[:, i]
            target_i = targets_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average dice loss
        dice_loss = 1 - torch.stack(dice_scores).mean()
        return dice_loss


def calculate_iou(predictions, targets, num_classes, ignore_index=255):
    """
    Calculate IoU (Intersection over Union) for each class
    """
    ious = []
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Ignore certain classes
    valid_mask = targets != ignore_index
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    for cls in range(num_classes):
        pred_cls = predictions == cls
        target_cls = targets == cls
        
        if target_cls.sum() == 0:
            # Class not present in ground truth
            if pred_cls.sum() == 0:
                ious.append(1.0)  # True negative
            else:
                ious.append(0.0)  # False positive
        else:
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            ious.append((intersection / union).item())
    
    return ious


def calculate_pixel_accuracy(predictions, targets, ignore_index=255):
    """
    Calculate pixel accuracy
    """
    valid_mask = targets != ignore_index
    correct = (predictions == targets) & valid_mask
    total = valid_mask.sum()
    
    if total == 0:
        return 0.0
    
    return (correct.sum().float() / total.float()).item()


def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    """
    Train model for one epoch
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_acc = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        predictions = torch.argmax(outputs, dim=1)
        batch_ious = calculate_iou(predictions, targets, num_classes)
        batch_acc = calculate_pixel_accuracy(predictions, targets)
        
        running_loss += loss.item()
        running_iou += np.mean(batch_ious)
        running_acc += batch_acc
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'mIoU': f'{np.mean(batch_ious):.4f}',
            'Acc': f'{batch_acc:.4f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_iou = running_iou / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    
    return epoch_loss, epoch_iou, epoch_acc


def validate_epoch(model, val_loader, criterion, device, num_classes):
    """
    Validate model for one epoch
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_acc = 0.0
    
    all_ious = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            batch_ious = calculate_iou(predictions, targets, num_classes)
            batch_acc = calculate_pixel_accuracy(predictions, targets)
            
            running_loss += loss.item()
            running_iou += np.mean(batch_ious)
            running_acc += batch_acc
            all_ious.extend(batch_ious)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{np.mean(batch_ious):.4f}',
                'Acc': f'{batch_acc:.4f}'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_iou = running_iou / len(val_loader)
    epoch_acc = running_acc / len(val_loader)
    
    # Calculate per-class IoU
    per_class_iou = []
    for cls in range(num_classes):
        class_ious = [all_ious[i] for i in range(cls, len(all_ious), num_classes)]
        per_class_iou.append(np.mean(class_ious))
    
    return epoch_loss, epoch_iou, epoch_acc, per_class_iou


def train_model(model, train_loader, val_loader, num_epochs, device, num_classes, 
                learning_rate=1e-4, weight_decay=1e-4, save_path='best_model.pth'):
    """
    Complete training loop
    """
    # Setup
    criterion = SegmentationLoss(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training history
    train_losses = []
    train_ious = []
    train_accs = []
    val_losses = []
    val_ious = []
    val_accs = []
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training
        train_loss, train_iou, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        
        # Validation
        val_loss, val_iou, val_acc, per_class_iou = validate_epoch(
            model, val_loader, criterion, device, num_classes
        )
        
        # Update learning rate
        scheduler.step(val_iou)
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, save_path)
            print(f'New best model saved with mIoU: {best_iou:.4f}')
        
        # Store history
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return {
        'train_losses': train_losses,
        'train_ious': train_ious,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'val_accs': val_accs,
        'best_iou': best_iou
    }


def plot_training_history(history, save_path=None):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_losses'], label='Train Loss')
    axes[0, 0].plot(history['val_losses'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['train_ious'], label='Train mIoU')
    axes[0, 1].plot(history['val_ious'], label='Val mIoU')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy
    axes[1, 0].plot(history['train_accs'], label='Train Acc')
    axes[1, 0].plot(history['val_accs'], label='Val Acc')
    axes[1, 0].set_title('Pixel Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Summary
    axes[1, 1].text(0.1, 0.7, f"Best Val mIoU: {max(history['val_ious']):.4f}", fontsize=12)
    axes[1, 1].text(0.1, 0.5, f"Final Train mIoU: {history['train_ious'][-1]:.4f}", fontsize=12)
    axes[1, 1].text(0.1, 0.3, f"Final Val mIoU: {history['val_ious'][-1]:.4f}", fontsize=12)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(model, val_loader, device, num_samples=8, class_names=None):
    """
    Visualize model predictions
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Take first image from batch
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            target = targets[0].cpu().numpy()
            pred = predictions[0].cpu().numpy()
            
            # Plot
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target, cmap='tab20')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='tab20')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show() 