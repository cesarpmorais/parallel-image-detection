#!/usr/bin/env python3
"""
Extrai buffers do modelo (running_mean, running_var do BatchNorm)
"""

import os
import torch
from torchvision.models import ResNet18_Weights
import torchvision.models as models

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

os.makedirs('../../weights', exist_ok=True)

print("Extraindo buffers do modelo...\n")

for name, buffer in model.named_buffers():
    # Ignorar num_batches_tracked (não usado em inferência)
    if 'num_batches_tracked' in name:
        continue
    
    data = buffer.detach().cpu().numpy()
    
    # Limpar nome para arquivo
    filename = name.replace('.', '_') + '.bin'
    filepath = os.path.join('../../weights', filename)
    
    # Salvar binário
    data.tofile(filepath)
    
    print(f"Saved: {filename}")
    print(f"  Shape: {data.shape}")
    print(f"  Size: {data.nbytes / 1024:.2f} KB")

print("\n[OK] Todos os buffers extraidos!")

