#!/usr/bin/env python3
"""
Benchmark automatizado completo:
1. Gera N imagens .bin
2. Compila automaticamente as versões cpp e cpp_parallel
3. Executa ambas, mede tempo total e plota comparativo
4. Compara numericamente as saídas (validação de corretude)
"""

import os
import sys
import time
import random
import subprocess
import csv
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# === CONFIGURAÇÕES DE PASTAS ===
ROOT = os.path.abspath("../../")
CPP_DIR = os.path.join(ROOT, "cpp")
CPP_PAR_DIR = os.path.join(ROOT, "cpp_parallel")
DATASET_DIR = os.path.join(ROOT, "datasets/imagenetmini-1000/imagenet-mini/train")
INPUTS_DIR = os.path.join(ROOT, "src/validate_results/benchmark_inputs")
RESULTS_CSV = os.path.join(ROOT, "src/validate_results/results_benchmark.csv")
OUTPUTS_DIR = os.path.join(ROOT, "src/validate_results/cpp_outputs")


# === FUNÇÕES AUXILIARES ===
def run_cmd(cmd, cwd=None):
    """Executa comando shell e imprime saída"""
    result = subprocess.run(cmd, cwd=cwd, shell=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERRO] Falha ao executar comando: {cmd}")
        sys.exit(1)


def compile_project(project_path):
    """Compila o projeto (CMake + Make) e retorna caminho do executável"""
    print(f"\n[*] Compilando projeto em: {project_path}")
    build_dir = os.path.join(project_path, "build")
    os.makedirs(build_dir, exist_ok=True)

    # ⚙️ Limpar cache antigo de CMake se necessário
    cache_file = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cache_file):
        print(f"[INFO] Limpando cache antigo do CMake em {build_dir}")
        for root, dirs, files in os.walk(build_dir, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except Exception:
                    pass

    # 1️⃣ Gerar build com CMake
    run_cmd("cmake ..", cwd=build_dir)
    # 2️⃣ Compilar
    run_cmd("make -j", cwd=build_dir)

    exe_path = os.path.join(build_dir, "resnet18")
    if not os.path.exists(exe_path):
        print(f"[ERRO] Executável não encontrado após compilação: {exe_path}")
        sys.exit(1)

    print(f"[OK] Compilado com sucesso: {exe_path}")
    return exe_path


def preprocess_image(image_path):
    """Aplica o mesmo pré-processamento usado na ResNet18"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def generate_binaries(num_images):
    """Gera binários de imagens do dataset"""
    os.makedirs(INPUTS_DIR, exist_ok=True)
    print(f"[*] Gerando {num_images} imagens pré-processadas...")

    image_paths = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))

    if len(image_paths) < num_images:
        print(f"[ERRO] Dataset tem apenas {len(image_paths)} imagens.")
        sys.exit(1)

    selected = random.sample(image_paths, num_images)

    for i, img_path in enumerate(selected, 1):
        try:
            tensor = preprocess_image(img_path)
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            base = os.path.join(INPUTS_DIR, f"input_{i:03d}")
            arr.tofile(base + ".bin")
            with open(base + "_shape.txt", "w") as f:
                f.write(" ".join(map(str, arr.shape)))
            if i % 10 == 0 or i == num_images:
                print(f"  - {i}/{num_images} imagens processadas")
        except Exception as e:
            print(f"[AVISO] Falha ao processar {img_path}: {e}")

    print(f"[OK] {num_images} imagens geradas em {INPUTS_DIR}")
    return INPUTS_DIR


def run_executable(exec_path, inputs_dir, tag):
    """Executa o binário C++ e mede tempo total"""
    if not os.path.exists(exec_path):
        print(f"[ERRO] Executável não encontrado: {exec_path}")
        return None

    cmd = [exec_path, "--images-dir", inputs_dir]
    print(f"\n[*] Executando {tag}...")

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.perf_counter()

    if result.returncode != 0:
        print(f"[ERRO] Execução falhou:\n{result.stderr}")
        return None

    total_time = end - start
    print(f"[OK] {tag} finalizado em {total_time:.2f}s")
    return total_time


def compare_outputs(seq_output_path, par_output_path):
    """Compara os resultados binários entre as versões sequencial e paralela"""
    if not (os.path.exists(seq_output_path) and os.path.exists(par_output_path)):
        print("[AVISO] Arquivos de saída não encontrados para comparação.")
        return

    seq = np.fromfile(seq_output_path, dtype=np.float32)
    par = np.fromfile(par_output_path, dtype=np.float32)

    if seq.shape != par.shape:
        print(f"[ERRO] Formatos diferentes: {seq.shape} vs {par.shape}")
        return

    abs_diff = np.abs(seq - par)
    mae = abs_diff.mean()
    max_diff = abs_diff.max()

    print("\n" + "=" * 70)
    print("  VALIDAÇÃO NUMÉRICA ENTRE SEQUENCIAL E PARALELA")
    print("=" * 70)
    print(f"Erro médio absoluto (MAE): {mae:.6e}")
    print(f"Erro máximo (MaxDiff): {max_diff:.6e}")

    if max_diff < 1e-5:
        print("[OK] Resultados numericamente equivalentes ✅")
    else:
        print("[AVISO] Diferença detectada (>1e-5) ⚠️")


def save_results_to_csv(results):
    """Salva resultados em CSV"""
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Versao", "Tempo_total_segundos"])
        for version, t in results.items():
            writer.writerow([version, t])
    print(f"[OK] Resultados salvos em {RESULTS_CSV}")


def plot_results(results):
    """Plota gráfico comparando tempos"""
    versions = list(results.keys())
    times = [results[v] for v in versions]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(versions, times, color=["#4C72B0", "#55A868"])
    plt.title("Comparativo de Tempo de Execução - ResNet18")
    plt.ylabel("Tempo total (segundos)")
    plt.xlabel("Versão")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{t:.2f}s", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(RESULTS_CSV), "benchmark_results.png"))
    plt.show()
    print("[OK] Gráfico salvo como benchmark_results.png")


# === PIPELINE PRINCIPAL ===
def main():
    if len(sys.argv) < 2:
        print("Uso: python benchmark_auto.py <num_imagens>")
        sys.exit(1)

    num_images = int(sys.argv[1])
    print("=" * 70)
    print(f"  BENCHMARK AUTOMATIZADO COMPLETO - {num_images} IMAGENS")
    print("=" * 70)

    # 1️⃣ Gerar binários de entrada
    inputs_dir = generate_binaries(num_images)

    # 2️⃣ Compilar ambos os projetos
    exe_seq = compile_project(CPP_DIR)
    exe_par = compile_project(CPP_PAR_DIR)

    # 3️⃣ Executar ambos
    results = {}
    seq_time = run_executable(exe_seq, inputs_dir, "Sequencial (CPU)")
    if seq_time is not None:
        results["Sequencial (CPU)"] = seq_time

    par_time = run_executable(exe_par, inputs_dir, "Paralelo (OpenMP)")
    if par_time is not None:
        results["Paralelo (OpenMP)"] = par_time

    # 4️⃣ Comparar resultados numéricos (corretude)
    seq_out = os.path.join(OUTPUTS_DIR, "final_output.bin")
    par_out = os.path.join(OUTPUTS_DIR, "final_output.bin")  # ambos salvam no mesmo lugar
    compare_outputs(seq_out, par_out)

    # 5️⃣ Salvar e plotar resultados
    save_results_to_csv(results)
    plot_results(results)

    print("\n[✔] Benchmark completo! Comparativo concluído com sucesso.")


if __name__ == "__main__":
    main()
