import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1x1 conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) + 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        
        # 1x1 conv
        feat1 = self.conv1x1(x)
        
        # Atrous convs
        atrous_feats = []
        for atrous_conv in self.atrous_convs:
            atrous_feats.append(atrous_conv(x))
        
        # Global average pooling
        feat_pool = self.global_avg_pool(x)
        feat_pool = F.interpolate(feat_pool, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate all features
        feats = [feat1] + atrous_feats + [feat_pool]
        out = torch.cat(feats, dim=1)
        out = self.project(out)
        
        return out


class DeepLabV3Plus(nn.Module):
    """DeepLab v3+ with ResNet backbone"""
    
    def __init__(self, num_classes=21, backbone='resnet50', pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        # Backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.low_level_channels = 256
            self.high_level_channels = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.low_level_channels = 256
            self.high_level_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Encoder (backbone)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # low-level features
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # high-level features
        
        # Modify layer3 and layer4 for dilated convolutions
        self._make_layer_dilated(self.layer3, stride=1, dilation=2)
        self._make_layer_dilated(self.layer4, stride=1, dilation=4)
        
        # ASPP
        self.aspp = ASPP(self.high_level_channels)
        
        # Decoder
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def _make_layer_dilated(self, layer, stride, dilation):
        """Modify layer to use dilated convolutions"""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (stride, stride)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)
                    
    def forward(self, x):
        size = x.shape[-2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        low_level_feat = self.layer1(x)  # 1/4 size
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)  # 1/8 size (due to dilation)
        
        # ASPP
        x = self.aspp(x)
        x = F.interpolate(x, size=(low_level_feat.shape[-2], low_level_feat.shape[-1]), 
                         mode='bilinear', align_corners=True)
        
        # Decoder
        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        return x


def get_deeplabv3plus_model(num_classes=21, backbone='resnet50', pretrained=True):
    """
    Factory function để tạo DeepLab v3+ model
    
    Args:
        num_classes (int): Số classes cho segmentation
        backbone (str): Backbone network ('resnet50' hoặc 'resnet101')
        pretrained (bool): Sử dụng pretrained weights
    
    Returns:
        DeepLabV3Plus model
    """
    return DeepLabV3Plus(num_classes=num_classes, backbone=backbone, pretrained=pretrained)


if __name__ == "__main__":
    # Test model
    model = get_deeplabv3plus_model(num_classes=21, backbone='resnet50')
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}") 