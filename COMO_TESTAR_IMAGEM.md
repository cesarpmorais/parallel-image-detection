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

