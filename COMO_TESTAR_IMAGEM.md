# 🖼️ Como Testar com Imagem Real

## Resumo: A Rede Já Identifica Imagens!

Sua rede ResNet18 C++ já consegue identificar objetos em imagens. Ela gera predições de 1000 classes do ImageNet.

## 🎯 O que a Rede Faz

1. **Recebe uma imagem** (224×224 pixels, RGB)
2. **Processa através de todas as camadas** (Conv2D, BatchNorm, ReLU, etc.)
3. **Gera 1000 scores** (um para cada classe do ImageNet)
4. **Você pode ver as top-5 predições** com probabilidades

## 📋 Como Testar com Imagem Real

### Teste com Sua Própria Imagem

**Passo 1:** Coloque sua imagem em `datasets/` (ex: `minha_imagem.jpg`)

**Passo 2:** Processar imagem
```python
# Criar script ou usar test_with_image.py
python test_with_image.py ../../datasets/minha_imagem.jpeg
```

**Passo 3:** Executar C++ e ver resultados

---

## 📊 Exemplo de Saída

```
======================================================================
  TOP 5 PREDICOES
======================================================================

  Rank   Classe   Probabilidade   Nome da Classe                                    
  ------ -------- --------------- --------------------------------------------------
  1      844               15.24% switch                                            
  2      662                8.84% modem                                             
  3      530                6.76% digital clock                                     
  4      620                3.26% laptop                                          
  5      446                3.01% binder                                            

======================================================================
  PREDICAO PRINCIPAL
======================================================================

  Classe: 844
  Nome: switch
  Confianca: 15.24%
```

---

## 🔍 O que Significa

- **Classe 844 = "switch"**: A rede identificou como "switch" (interruptor)
- **Confiança 15.24%**: Probabilidade de ser essa classe
- **Top-5**: As 5 classes mais prováveis

**Nota:** Os dados de teste atuais são sintéticos (não uma imagem real), por isso as predições podem não fazer muito sentido. Para testar com imagem real, use `test_cavalo.py`.

---

## 📈 Benchmark com Múltiplas Imagens

Para benchmarking em larga escala com validação automática, use o script `benchmark.py`:

### Opção 1: Executar a partir do diretório do projeto (raiz)

```bash
# Validar 5 imagens e gerar relatório
python src/validate_results/benchmark.py \
  --bin cpp/build/resnet18 \
  --images datasets \
  --out benchmark_results.csv \
  --max-images 5 \
  --validate
```

### Parâmetros do benchmark.py

```
--bin <path>          Caminho do executável C++ (padrão: procura automaticamente)
--images <path>       Diretório de imagens (padrão: datasets/)
--out <path>          Arquivo CSV de saída (padrão: benchmark_results.csv)
--max-images <n>      Limite de imagens a processar (0 = todas)
-n <n>                Alias para --max-images
--repeat <n>          Número de repetições por imagem (padrão: 1)
--validate            Validar predições contra modelo PyTorch de referência
--verbose             Mostrar informações de debug
--timeout <s>         Timeout em segundos (padrão: 60)
```

### Exemplo de Saída

```
Found 5 images. Creating temp preprocessed inputs...
Running C++ binary on 5 preprocessed images (--repeat 1)...
C++ binary completed (wall time: 23583.0 ms)
Results written to benchmark_validated.csv

=== Validation Results ===
Passed: 5/5
  n01440764_tench.JPEG                     ✓ PASS     (ref=0, cpp=0)
  n01443537_goldfish.JPEG                  ✓ PASS     (ref=1, cpp=1)
  n01484850_great_white_shark.JPEG         ✓ PASS     (ref=2, cpp=2)
  n01491361_tiger_shark.JPEG               ✓ PASS     (ref=3, cpp=3)
  n01494475_hammerhead.JPEG                ✓ PASS     (ref=842, cpp=842)
```

### O que o Benchmark Faz

1. **Preprocessa imagens** em formato `.bin` (normalização ImageNet)
2. **Executa uma única vez** o binário C++ com todas as imagens
3. **Coleta timings por camada** para análise de desempenho
4. **Valida predições** (opcional) comparando contra PyTorch
5. **Gera CSV** com resultados por imagem e por camada

### Arquivo CSV de Saída

O arquivo CSV contém:
- `image`: Nome da imagem
- `top1`: Classe prevista pelo C++
- `valid`: Se a predição está correta (quando `--validate` ativado)
- `layer_conv1`, `layer_bn1`, ..., `layer_total`: Timings por camada em ms

Isso permite análise detalhada de performance e verificação de correção.

