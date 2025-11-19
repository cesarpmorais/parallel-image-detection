# 🧩 Progresso do Projeto — Classificação de Imagens com ResNet-18 (C++)

## 🎯 **Objetivo Geral**

Implementar, validar e comparar o desempenho da rede **ResNet-18** em **C++**,  
avaliando as versões **sequencial (CPU)** e **paralela (CPU com OpenMP e GPU com CUDA)**,  
com ênfase em **corretude numérica** e **ganho de desempenho**.

---

## ✅ **Etapas já concluídas**

### 🧱 1. Implementação Sequencial em C++
- **Descrição:** Implementação completa da arquitetura ResNet-18 em C++ puro.  
- **Arquivos:**  
  `conv2d.cpp`, `batchnorm.cpp`, `relu.cpp`, `maxpool.cpp`,  
  `basicblock.cpp`, `adaptiveavgpool.cpp`, `linear.cpp`, `tensor.cpp`, `main.cpp`.  
- **Status:** ✅ Concluído  
- **Resultado:** Rede funcional executando inferência sobre imagens processadas.

---

### 🧠 2. Validação Numérica com PyTorch
- **Descrição:** Comparação entre as saídas do modelo C++ e o modelo original PyTorch.  
- **Scripts:**  
  `reference_model.py`, `validate.py`, `test_with_image.py`.  
- **Status:** ✅ Concluído  
- **Resultado:** Corretude confirmada camada a camada e na saída final (erro máximo abaixo de `1e-5`).

---

### 🧾 3. Geração e Pré-processamento de Dados
- **Descrição:** Conversão de imagens do ImageNet-mini em tensores binários (`.bin`) com normalização e reshape.  
- **Scripts:**  
  `test_with_image.py`, `benchmark_auto.py`.  
- **Status:** ✅ Concluído  
- **Resultado:** Pipeline de conversão automática pronto e integrado aos testes.

---

### ⚙️ 4. Execução Automática e Validação Completa
- **Descrição:** Automação do fluxo de inferência — geração de input, execução C++, carregamento de saída e validação com PyTorch.  
- **Script:** `test_with_image.py` (com execução automática via `subprocess`).  
- **Status:** ✅ Concluído  
- **Resultado:** Execução completa em um único comando, sem intervenção manual.

---

### 🧩 5. Paralelização em CPU (OpenMP)
- **Descrição:** Inserção de diretivas `#pragma omp parallel for` nas operações mais custosas (ex.: convoluções e blocos residuais).  
- **Local:**  
  `cpp_parallel/` (cópia modificada da implementação sequencial).  
- **Status:** ✅ Concluído (versão inicial).  
- **Resultado:** Aceleração observável em testes com múltiplos núcleos.

---

### 📈 6. Benchmark Automatizado e Comparativo
- **Descrição:** Criação de um sistema de benchmark completo que:
  1. Gera automaticamente `N` imagens `.bin` de entrada,  
  2. Compila ambos os projetos (`cpp` e `cpp_parallel`),  
  3. Executa ambos medindo o tempo total,  
  4. Valida a corretude numérica (`MAE`, `MaxDiff`),  
  5. Gera gráfico comparativo de desempenho.  
- **Script:** `benchmark_parallel_vs_sequential.py`.  
- **Exemplo exec.:** `python benchmark_parallel_vs_sequential.py 10`.  
- **Status:** ✅ Concluído.  
- **Resultado:** Pipeline 100% automatizado, com resultados reprodutíveis e comparáveis.

---

## ⏳ **Etapas pendentes / em desenvolvimento**

### ⚡ 7. Paralelização em GPU (CUDA)
- **Descrição:** Migrar camadas intensivas (Conv2D, BatchNorm, Linear) para CUDA, criando um diretório `cpp_cuda/`.  
- **Objetivo:** Explorar paralelismo massivo e comparar com CPU + OpenMP.  
- **Status:** ⏳ Em planejamento.  
- **Próximos passos:**  
  - Criar kernels CUDA (`.cu`) para convolução e multiplicação de matrizes.  
  - Integrar com `benchmark_auto.py`.  

---

### 📊 8. Benchmark CPU × GPU
- **Descrição:** Expandir o benchmark atual para incluir o executável CUDA.  
- **Objetivo:** Comparar desempenho entre três implementações (CPU, OpenMP e CUDA).  
- **Status:** ⏳ Pendente.  
- **Próximos passos:**  
  - Adicionar caminho `CPP_CUDA_DIR` ao script.  
  - Gerar gráfico com as três barras e tempos médios.  

---

### 🧾 9. Relatório e Resultados Experimentais
- **Descrição:** Compilar todos os resultados numéricos e de desempenho no modelo SBC (Overleaf).  
- **Conteúdo:**  
  - Seções: *Metodologia, Paralelização Proposta, Resultados e Discussão*.  
  - Inclusão de tabelas (tempo, speedup, erro) e gráficos gerados.  
- **Status:** ⚙️ Em andamento.  

---

### 🧪 10. Comparação Camada a Camada (opcional)
- **Descrição:** Comparar ativações intermediárias (`after_conv1`, `after_bn1`, etc.) entre versões sequencial, OpenMP e CUDA.  
- **Status:** ⚙️ Opcional (para estudo de precisão e estabilidade numérica).  

---

## 🧭 **Resumo de Progresso**

| Etapa | Título | Status |
|-------|---------|--------|
| 1 | Implementação sequencial (CPU) | ✅ Concluído |
| 2 | Validação numérica com PyTorch | ✅ Concluído |
| 3 | Geração e pré-processamento de imagens | ✅ Concluído |
| 4 | Execução e validação automatizada | ✅ Concluído |
| 5 | Paralelização em CPU (OpenMP) | ✅ Concluído |
| 6 | Benchmark automatizado + gráfico + validação | ✅ Concluído |
| 7 | Paralelização em GPU (CUDA) | ⏳ Planejado |
| 8 | Benchmark CPU × GPU | ⏳ Planejado |
| 9 | Relatório SBC (metodologia e resultados) | ⚙️ Em andamento |
| 10 | Comparação camada a camada (opcional) | ⚙️ Em análise |

---

## 🧠 **Situação Atual**

O projeto já possui:
- Implementação e validação numérica confiável da ResNet-18 em C++;  
- Paralelização via OpenMP com corretude garantida;  
- Pipeline automatizado para geração, execução, validação e análise;  

O próximo passo natural é **migrar para GPU (CUDA)** e **comparar ganhos de desempenho reais**.
