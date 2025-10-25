# Resnet18 architecture

```
INPUT (3x224x224)
    ↓
conv1: 3→64, 7x7, stride=2, padding=3
bn1 + relu
maxpool: 3x3, stride=2
    ↓
layer1: 2x BasicBlock (64→64)
    ↓
layer2: 2x BasicBlock (64→128, com downsample)
    ↓
layer3: 2x BasicBlock (128→256, com downsample)
    ↓
layer4: 2x BasicBlock (256→512, com downsample)
    ↓
avgpool: adaptative (reduz para 1x1)
fc: 512→1000 classes
    ↓
OUTPUT (1000 classes)
```