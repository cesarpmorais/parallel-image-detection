# Guia de Implementação ResNet-18

## Visão Geral
Este documento descreve a estratégia de implementação para ResNet-18 em C++ para permitir paralelização customizada. A implementação foca na modularidade, permitindo que cada tipo de camada seja otimizada independentemente.

## Componentes da Arquitetura

### Tipos de Camadas Necessárias
A arquitetura ResNet-18 requer implementar cinco tipos fundamentais de camadas:

#### 1. Camada Convolucional (Conv2D)
- Realiza convolução 2D com filtros treináveis
- Suporta tamanhos de kernel variáveis (1x1, 3x3, 7x7)
- Stride e padding configuráveis
- Sem termo de bias (absorvido pelo BatchNorm)

#### 2. Normalização em Lotes (BatchNorm2D)
- Normaliza ativações usando estatísticas móveis
- Aplica parâmetros de escala (gamma) e deslocamento (beta) treináveis
- Pode ser fundida com convolução precedente para inferência

#### 3. Função de Ativação (ReLU)
- Operação elemento por elemento: max(0, x)
- Aplicada in-place para economizar memória

#### 4. Camadas de Pooling
- **MaxPool2D**: reduz dimensões espaciais tomando o máximo em cada janela
- **AdaptiveAvgPool2D**: calcula média das dimensões espaciais para tamanho fixo de saída (1x1)

#### 5. Camada Totalmente Conectada (Linear)
- Multiplicação de matrizes para classificação final
- Mapeia 512 características para 1000 pontuações de classe

## Layout de Dados

### Organização da Memória

#### Mapas de Características (Ativações)
- Armazenar como tensores 4D: [batch, channels, height, width]
- Layout de memória contígua para acesso eficiente
- Exemplo: saída da layer1 é [1, 64, 56, 56] = 200.704 floats

#### Pesos
- Pesos Conv2D: [out_channels, in_channels, kernel_h, kernel_w]
- Parâmetros BatchNorm: [channels] para gamma, beta, mean, variance
- Pesos Linear: [out_features, in_features]

#### Gerenciamento de Buffer
- Pré-alocar buffers para ativações intermediárias
- Reutilizar buffers onde as dependências permitirem
- Exemplo: após layer1 completar, seu buffer de entrada pode ser reutilizado para layer2

## Estrutura de Classes

### Classes de Camada Principais

#### Classe Conv2D
- Armazena tensor de pesos e configuração da camada
- Método `forward()` realiza operação de convolução
- `load_weights()` lê arquivos de pesos binários
- Metadados: in_channels, out_channels, kernel_size, stride, padding

#### Classe BatchNorm2D
- Armazena gamma, beta, running_mean, running_variance
- Para inferência: `output = gamma * (input - mean) / sqrt(variance + eps) + beta`
- Pode ser fundida: mesclar na conv anterior como `weight_fused = weight * gamma / sqrt(variance)`, `bias_fused = beta - mean * gamma / sqrt(variance)`

#### Classe BasicBlock
- Encapsula estrutura do bloco residual
- Contém duas sequências Conv2D + BatchNorm + ReLU
- Implementa conexão de salto (identity ou downsampling)
- `forward()` gerencia tanto o caminho principal quanto o residual

#### Classe ResNet18
- Orquestrador de rede de nível superior
- Possui todas as instâncias de camadas
- `forward()` encadeia execuções de camadas
- `load_model()` inicializa todos os pesos dos arquivos

## Pipeline de Passagem Direta

### Fluxo de Execução

#### Estágio 1: Processamento Inicial
- Imagem de entrada (3×224×224) entra na conv1
- Conv1 produz 64 mapas de características na resolução 112×112 (stride=2)
- BatchNorm1 normaliza, ReLU ativa
- MaxPool reduz para 56×56 (stride=2)

#### Estágio 2-5: Camadas Residuais
- Cada camada contém 2 BasicBlocks
- Layer2, 3, 4 primeiros blocos incluem downsampling (stride=2 + conv 1×1 no caminho de salto)
- Dimensões espaciais: 56×56 → 28×28 → 14×14 → 7×7
- Canais: 64 → 128 → 256 → 512

#### Estágio 6: Classificação
- AdaptiveAvgPool reduz 7×7 para 1×1
- Camada linear mapeia 512 características para 1000 pontuações de classe
- Softmax (opcional) para probabilidades

## Detalhes de Implementação do BasicBlock

### Estrutura
Cada BasicBlock realiza:
```
input → conv1(3×3) → bn1 → relu → conv2(3×3) → bn2 → (+residual) → relu → output
```
- **Caminho Principal**: duas convoluções 3×3 com BatchNorm entre elas
- **Caminho de Salto**: identidade (mesmas dimensões) ou conv 1×1 + BatchNorm (mudança de dimensão)

### Lógica de Downsampling

#### Sem Downsampling (blocos layer1, segundos blocos de layer2/3/4):
- Conexão de salto é cópia direta da entrada
- Dimensões de saída correspondem às dimensões de entrada

#### Com Downsampling (primeiros blocos de layer2/3/4):
- Caminho principal: primeira conv usa stride=2
- Caminho de salto: conv 1×1 com stride=2 ajusta canais e tamanho espacial
- Exemplo: layer2.0 transforma 64×56×56 → 128×28×28

## Estratégia de Carregamento de Pesos

### Organização de Arquivos
Pesos exportados do PyTorch são armazenados como arquivos binários planos (.bin):
- Um arquivo por tensor de parâmetro
- Ponto flutuante de 32 bits (4 bytes por valor)
- Ordem row-major (contígua estilo C)

### Processo de Carregamento

#### Fase de Inicialização:
- Abrir arquivo binário para leitura
- Ler metadados (forma) de JSON separado ou hardcode da arquitetura
- Alocar buffer com tamanho correto
- Ler bytes diretamente no buffer
- Verificar se o tamanho corresponde ao esperado

#### Validação:
- Comparar tamanho do arquivo com tamanho esperado do tensor
- Opcionalmente computar estatísticas (min, max, média) para verificação
- Testar saída da primeira camada contra referência

## Requisitos de Memória

### Memória de Ativações
Maiores ativações ocorrem no início da rede:
- Após conv1: 1×64×112×112 = 802.816 floats (3,2 MB)
- Após layer1: 1×64×56×56 = 200.704 floats (0,8 MB)
- Pico de memória: ~10-15 MB para inferência de imagem única

### Memória de Pesos
Total de parâmetros: ~11,7M
- Camadas convolucionais: ~11,2M parâmetros (~45 MB)
- Parâmetros BatchNorm: ~0,1M parâmetros (~0,4 MB)
- Totalmente conectada: ~0,5M parâmetros (~2 MB)
- **Total: ~48 MB**

## Custo Computacional

### Contagens de Operações (FLOPs)
Convoluções dominam: ~95% do compute
- Conv1 (7×7): 118M FLOPs
- Layer1-4: ~1,7B FLOPs total
- Totalmente conectada: 0,5M FLOPs

### Detalhamento por camada:
- BasicBlocks com convs 3×3: ~150-600M FLOPs cada
- Downsampling convs 1×1: ~10-50M FLOPs cada
- Pooling/ativação: negligível (<1% total)

## Ordem de Implementação

### Sequência de Desenvolvimento Recomendada

#### Fase 1: Camadas Principais
1. Implementar Conv2D (versão naive primeiro)
2. Implementar ReLU (trivial)
3. Testar saída conv1 contra referência PyTorch

#### Fase 2: Camadas de Apoio
4. Implementar BatchNorm (pode pular inicialmente - usar pesos fundidos)
5. Implementar MaxPool2D
6. Implementar AdaptiveAvgPool2D

#### Fase 3: Blocos
7. Implementar BasicBlock sem conexão de salto
8. Adicionar conexão de salto identidade
9. Adicionar conexão de salto downsampling

#### Fase 4: Rede Completa
10. Implementar classe ResNet18
11. Encadear todas as camadas
12. Validar saída end-to-end

#### Fase 5: Otimização
13. Otimizar camadas críticas (Conv2D)
14. Adicionar estratégias de reutilização de memória
15. Perfilar e ajustar gargalos

## Estratégia de Teste

### Abordagem de Validação

#### Teste Camada por Camada:
- Exportar ativações intermediárias do PyTorch
- Comparar saída da camada C++ com referência
- Tolerância: diferença absoluta < 1e-4 (considerando diferenças de ponto flutuante)

#### Teste End-to-End:
- Usar entrada aleatória fixa
- Comparar saída final de 1000 classes
- Verificar se predições top-5 coincidem

#### Estabilidade Numérica:
- Testar com casos extremos (zeros, valores grandes)
- Verificar que não há valores NaN ou Inf
- Verificar gradiente de erros entre camadas

## Opções de Simplificação

### Otimizações Opcionais para Implementação Inicial

#### Fusão BatchNorm:
- Pré-computar pesos conv fundidos offline
- Elimina camadas BatchNorm na inferência
- Reduz leituras de memória e operações aritméticas

#### Precisão Reduzida:
- Começar com float32 para correção
- Depois experimentar com float16 (se hardware suportar)

#### Implementação de Subconjunto:
- Inicialmente implementar apenas até layer2
- Validar correção da arquitetura com menos camadas
- Expandir para rede completa uma vez confiante

#### Tamanho de Batch Fixo:
- Otimizar para batch_size=1 (inferência de imagem única)
- Simplifica gerenciamento de buffer
- Suficiente para maioria dos pipelines de detecção de objetos