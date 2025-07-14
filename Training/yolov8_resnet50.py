# yolov8_resnet50.py - Integration with YOLOv8
import torch
import torch.nn as nn
from resnet50 import create_resnet50_backbone

class YOLOv8ResNet50(nn.Module):
    """YOLOv8 with ResNet50 backbone"""
    
    def __init__(self, num_classes=80, input_channels=3):
        super().__init__()
        self.backbone = create_resnet50_backbone(input_channels=input_channels)
        
        # Feature pyramid network (simplified)
        self.fpn = self._build_fpn()
        
        # YOLO heads (simplified - you'll need full YOLOv8 head implementation)
        self.heads = self._build_heads(num_classes)
    
    def _build_fpn(self):
        """Build Feature Pyramid Network"""
        return nn.ModuleDict({
            'lateral_256': nn.Conv2d(256, 256, 1),
            'lateral_512': nn.Conv2d(512, 256, 1),
            'lateral_1024': nn.Conv2d(1024, 256, 1),
            'lateral_2048': nn.Conv2d(2048, 256, 1),
            'upsample': nn.Upsample(scale_factor=2, mode='nearest'),
        })
    
    def _build_heads(self, num_classes):
        """Build YOLO detection heads"""
        return nn.ModuleList([
            nn.Conv2d(256, (num_classes + 5) * 3, 1),  # 3 anchors per scale
            nn.Conv2d(256, (num_classes + 5) * 3, 1),
            nn.Conv2d(256, (num_classes + 5) * 3, 1),
        ])
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone.get_feature_maps(x)
        
        # Apply FPN (simplified)
        p4 = self.fpn['lateral_2048'](features['layer4'])
        p3 = self.fpn['lateral_1024'](features['layer3']) + self.fpn['upsample'](p4)
        p2 = self.fpn['lateral_512'](features['layer2']) + self.fpn['upsample'](p3)
        
        # Apply detection heads
        outputs = []
        for i, feat in enumerate([p2, p3, p4]):
            outputs.append(self.heads[i](feat))
        
        return outputs

# Usage example
def create_yolov8_resnet50(num_classes=80):
    """Create YOLOv8 with ResNet50 backbone"""
    return YOLOv8ResNet50(num_classes=num_classes)

# Test the integration
if __name__ == "__main__":
    model = create_yolov8_resnet50(num_classes=80)
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    print("YOLOv8-ResNet50 output shapes:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: {out.shape}")