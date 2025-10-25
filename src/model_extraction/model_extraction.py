import os
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import json

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

print("Estrutura do modelo:", model)

os.makedirs('weights', exist_ok=True)
for name, param in model.named_parameters():
    weight = param.detach().cpu().numpy()

    # Limpar nome para arquivo
    filename = name.replace('.', '_') + '.bin'
    filepath = os.path.join('weights', filename)

    # Salvar binário
    weight.tofile(filepath)

    # Log
    print(f"Saved: {filename}")
    print(f"  Shape: {weight.shape}")
    print(f"  Size: {weight.nbytes / 1024:.2f} KB")

print("\n✅ Todos os pesos extraídos!")
print(f"Total de arquivos: {len(list(model.parameters()))}")
