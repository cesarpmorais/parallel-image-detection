#!/usr/bin/env python3
"""
ResNet18 Reference Implementation with Intermediate Output Logging
For validating C++ implementation step-by-step
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os
from PIL import Image

class ResNet18Reference:
    def __init__(self, weights_dir='weights'):
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.weights_dir = weights_dir
        self.activations = {}
        
    def forward_with_logging(self, x, log_dir='reference_outputs'):
        """Forward pass with intermediate activation logging"""
        os.makedirs(log_dir, exist_ok=True)
        
        print("🔍 Forward pass with logging...")
        
        # Input
        self._save_tensor(x, f"{log_dir}/input.bin")
        print(f"Input: {x.shape}")
        
        # Conv1 + BN1 + ReLU + MaxPool
        x = self.model.conv1(x)
        self._save_tensor(x, f"{log_dir}/after_conv1.bin")
        print(f"After conv1: {x.shape}")
        
        x = self.model.bn1(x)
        self._save_tensor(x, f"{log_dir}/after_bn1.bin")
        print(f"After bn1: {x.shape}")
        
        x = torch.relu(x)
        self._save_tensor(x, f"{log_dir}/after_relu1.bin")
        print(f"After relu1: {x.shape}")
        
        x = self.model.maxpool(x)
        self._save_tensor(x, f"{log_dir}/after_maxpool.bin")
        print(f"After maxpool: {x.shape}")
        
        # Layer 1
        x = self.model.layer1(x)
        self._save_tensor(x, f"{log_dir}/after_layer1.bin")
        print(f"After layer1: {x.shape}")
        
        # Layer 2
        x = self.model.layer2(x)
        self._save_tensor(x, f"{log_dir}/after_layer2.bin")
        print(f"After layer2: {x.shape}")
        
        # Layer 3
        x = self.model.layer3(x)
        self._save_tensor(x, f"{log_dir}/after_layer3.bin")
        print(f"After layer3: {x.shape}")
        
        # Layer 4
        x = self.model.layer4(x)
        self._save_tensor(x, f"{log_dir}/after_layer4.bin")
        print(f"After layer4: {x.shape}")
        
        # AdaptiveAvgPool
        x = self.model.avgpool(x)
        self._save_tensor(x, f"{log_dir}/after_avgpool.bin")
        print(f"After avgpool: {x.shape}")
        
        # Flatten
        x = torch.flatten(x, 1)
        self._save_tensor(x, f"{log_dir}/after_flatten.bin")
        print(f"After flatten: {x.shape}")
        
        # FC
        x = self.model.fc(x)
        self._save_tensor(x, f"{log_dir}/final_output.bin")
        print(f"Final output: {x.shape}")
        
        return x
    
    def _save_tensor(self, tensor, filepath):
        """Save tensor as binary file"""
        data = tensor.detach().cpu().numpy().astype(np.float32)
        data.tofile(filepath)
        
        # Also save shape info
        shape_file = filepath.replace('.bin', '_shape.txt')
        with open(shape_file, 'w') as f:
            f.write(' '.join(map(str, data.shape)))

def preprocess_image(image_path):
    """Standard ImageNet preprocessing"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

if __name__ == "__main__":
    # Create reference model
    ref_model = ResNet18Reference()
    
    # Test with a sample image
    test_image_path = "dataset/test_image.jpg"  # You'll need to provide this
    
    if os.path.exists(test_image_path):
        print(f"📸 Processing: {test_image_path}")
        
        # Preprocess
        input_tensor = preprocess_image(test_image_path)
        
        # Forward pass with logging
        output = ref_model.forward_with_logging(input_tensor)
        
        # Show top predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        print("\n🎯 Top 5 predictions:")
        for i in range(5):
            print(f"  {top5_idx[i].item()}: {top5_prob[i].item():.4f}")
            
        print("\n✅ Reference outputs saved to reference_outputs/")
        
    else:
        print(f"❌ Test image not found: {test_image_path}")
        print("Please add a test image to dataset/test_image.jpg")
