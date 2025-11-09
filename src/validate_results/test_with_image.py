#!/usr/bin/env python3
"""
Teste Completo com Imagem Real
Processa imagem, executa C++, e mostra predições com nomes
"""

import numpy as np
import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms

def download_imagenet_labels():
    """Baixa labels do ImageNet"""
    import urllib.request
    
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_file = "imagenet_classes.txt"
    
    if not os.path.exists(labels_file):
        print(f"[*] Baixando labels do ImageNet...")
        try:
            urllib.request.urlretrieve(url, labels_file)
            print(f"[OK] Labels baixados!")
        except Exception as e:
            print(f"[AVISO] Nao foi possivel baixar labels: {e}")
            return None
    
    # Carregar labels
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    return labels

def preprocess_image(image_path):
    """Pré-processa imagem para ResNet18"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor

def save_input_for_cpp(tensor, output_dir="test_data"):
    """Salva tensor como binário para C++"""
    os.makedirs(output_dir, exist_ok=True)
    
    data = tensor.detach().cpu().numpy().astype(np.float32)
    data.tofile(os.path.join(output_dir, "input.bin"))
    
    with open(os.path.join(output_dir, "input_shape.txt"), 'w') as f:
        f.write(' '.join(map(str, data.shape)))
    
    print(f"[OK] Input salvo para C++: {data.shape}")

def softmax(x):
    """Softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def load_cpp_output():
    """Carrega saída do C++"""
    output_file = "cpp_outputs/final_output.bin"
    shape_file = "reference_outputs/final_output_shape.txt"
    
    if not os.path.exists(output_file):
        return None
    
    output = np.fromfile(output_file, dtype=np.float32)
    
    if os.path.exists(shape_file):
        with open(shape_file, 'r') as f:
            shape = tuple(map(int, f.read().strip().split()))
        output = output.reshape(shape)
    
    # Flatten se necessário
    if len(output.shape) == 4:
        output = output.flatten()
    elif len(output.shape) == 2:
        output = output[0]
    
    return output

def main():
    print("="*70)
    print("  TESTE COM IMAGEM REAL - ResNet18 C++")
    print("="*70)
    
    # Verificar se há imagem
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Tentar encontrar imagem
        possible_paths = [
            "../../datasets/cavalo.jpeg",
            "../datasets/cavalo.jpeg",
            "datasets/cavalo.jpeg"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
    
    if not image_path or not os.path.exists(image_path):
        print(f"\n[AVISO] Nenhuma imagem encontrada.")
        print(f"Uso: python test_with_image.py <caminho_imagem>")
        print(f"Ou coloque uma imagem em datasets/")
        print(f"\nTestando com dados de teste existentes...")
        
        # Testar com saída existente
        output = load_cpp_output()
        if output is None:
            print("[ERRO] Execute primeiro: cd ../../cpp && .\\resnet18.exe")
            return 1
        
        labels = download_imagenet_labels()
        
        probs = softmax(output)
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_probs = probs[top5_idx]
        
        print(f"\n[*] Top 5 Predicoes (dados de teste):")
        print(f"  {'Rank':<6} {'Classe':<8} {'Probabilidade':<15} {'Nome':<40}")
        print(f"  {'-'*6} {'-'*8} {'-'*15} {'-'*40}")
        
        for i, (idx, prob) in enumerate(zip(top5_idx, top5_probs), 1):
            name = labels[idx] if labels else f"Classe {idx}"
            print(f"  {i:<6} {idx:<8} {prob*100:>14.2f}% {name:<40}")
        
        return 0
    
    # Processar imagem real
    print(f"\n[1] Carregando imagem: {image_path}")
    try:
        input_tensor = preprocess_image(image_path)
        print(f"[OK] Imagem carregada: {input_tensor.shape}")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar: {e}")
        return 1
    
    # Salvar para C++
    print(f"\n[2] Salvando input para programa C++...")
    save_input_for_cpp(input_tensor)
    
    print(f"\n[3] Execute o programa C++ agora:")
    print(f"    cd ../../cpp")
    print(f"    .\\resnet18.exe")
    print(f"\n    Depois volte aqui e pressione Enter...")
    input()
    
    # Carregar saída do C++
    print(f"\n[4] Carregando saida do C++...")
    output = load_cpp_output()
    
    if output is None:
        print("[ERRO] Saida do C++ nao encontrada!")
        print("Execute: cd ../../cpp && .\\resnet18.exe")
        return 1
    
    # Carregar labels
    labels = download_imagenet_labels()
    
    # Calcular predições
    probs = softmax(output)
    top5_idx = np.argsort(probs)[::-1][:5]
    top5_probs = probs[top5_idx]
    
    # Mostrar resultados
    print(f"\n" + "="*70)
    print(f"  RESULTADOS DA CLASSIFICACAO")
    print("="*70)
    
    print(f"\n  Imagem: {image_path}")
    print(f"\n  Top 5 Predicoes:")
    print(f"  {'Rank':<6} {'Classe':<8} {'Probabilidade':<15} {'Nome':<40}")
    print(f"  {'-'*6} {'-'*8} {'-'*15} {'-'*40}")
    
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_probs), 1):
        name = labels[idx] if labels else f"Classe {idx}"
        print(f"  {i:<6} {idx:<8} {prob*100:>14.2f}% {name:<40}")
    
    print(f"\n  [*] Predicao Principal: {labels[top5_idx[0]] if labels else f'Classe {top5_idx[0]}'}")
    print(f"      Confianca: {top5_probs[0]*100:.2f}%")
    
    # Comparar com Python
    print(f"\n[5] Comparando com referencia Python...")
    try:
        from reference_model import ResNet18Reference
        
        ref_model = ResNet18Reference()
        with torch.no_grad():
            ref_output = ref_model.model(input_tensor)
        
        ref_probs = torch.nn.functional.softmax(ref_output[0], dim=0).numpy()
        ref_top5 = np.argsort(ref_probs)[::-1][:5]
        
        print(f"\n  Comparacao Top-5:")
        print(f"  {'Rank':<6} {'C++':<40} {'Python':<40} {'Match':<8}")
        print(f"  {'-'*6} {'-'*40} {'-'*40} {'-'*8}")
        
        matches = 0
        for i, (cpp_idx, py_idx) in enumerate(zip(top5_idx, ref_top5), 1):
            cpp_name = labels[cpp_idx] if labels else f"Classe {cpp_idx}"
            py_name = labels[py_idx] if labels else f"Classe {py_idx}"
            match = "SIM" if cpp_idx == py_idx else "NAO"
            if cpp_idx == py_idx:
                matches += 1
            print(f"  {i:<6} {cpp_name:<40} {py_name:<40} {match:<8}")
        
        print(f"\n  Top-5 Accuracy: {matches}/5 = {matches*100/5:.1f}%")
        
        if top5_idx[0] == ref_top5[0]:
            print(f"  [OK] Top-1 predicao CORRETA!")
        else:
            print(f"  [AVISO] Top-1 diferente")
            
    except Exception as e:
        print(f"  [AVISO] Nao foi possivel comparar: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

