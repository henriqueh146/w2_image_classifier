# Classificador Binário de Imagens - Cavalos x Humanos

Este projeto realiza a classificação binária de imagens para identificar se a imagem enviada representa um **cavalo** (`horse`) ou um **humano** (`human`). Ele inclui treinamento, avaliação e inferência via API REST.

---

## Justificativas Técnicas

### Bibliotecas e Frameworks

- **PyTorch Lightning**: abstrai boilerplate de treino e validação com foco em reprodutibilidade e organização.
- **Torchvision**: carregamento e transformações em imagens.
- **scikit-learn**: métricas de avaliação robustas (f1-score, acurárica e matriz de confusão).
- **FastAPI**: para servir inferência via REST.
- **Poetry**: gerenciamento moderno de dependências e ambientes virtuais.

### Arquitetura

- Modelo base: `ResNet18` pré-treinada (transfer learning).
  - Boa relação entre performance e tempo de treino.
  - Evita necessidade de milhares de imagens com poucos dados.
- Camada de saída com 1 unidade e ativação `sigmoid`, pois trata-se de um problema de classificação **binária**.

### Função Objetivo

- **Binary Cross-Entropy (BCE)**, a função padrão para classificação binária.

### Estratégia de Validação

- Separação explícita de diretórios `train/` e `validation/`.
- Uso de métricas `val_loss`, `val_acc` e `val_f1`.
- `EarlyStopping` com `ModelCheckpoint`, isto é, o melhor modelo é salvo nas iterações de treinamento para evitar overfitting e o processo pára antes da época final, caso o desempenho não melhore nas próximas 5 épocas.

---

## Setup do Ambiente

Requisitos: Python 3.10+

```bash
# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clonar o projeto. Os dados de treinamento e validação já estão disponíveis no próprio repositório por conveniência.
git clone https://github.com/henriqueh146/w2_image_classifier.git
cd w2_image_classifier

# Instalar dependências
poetry install

# Ativar o ambiente
poetry shell

# Ativar API
uvicorn src.api:app --reload

```

### Treinamento
```bash
python src/train.py
```

### Avaliação
```bash
python src/eval.py
```

### Inferência

1. Com a API ativada (conforme instrução acima), acessar a URL http://localhost:8000/docs no navegador.
2. Click em POST
3. Click em "Try it Out!"
4. Click em "Escolher Arquivo"
5. Selecionar imagem do dataset de validação
6. Click no botão "Execute"

Seguindo este passo a passo, a API retornará uma resposta no seguinte formato:

```bash
{
  "prediction": "Is a human"
}
```

Ou você pode acessar a API via curl, onde "path/para/imagem.png" deve ser substituído pelo path da imagem na sua máquina:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H  "accept: application/json" \
  -H  "Content-Type: multipart/form-data" \
  -F "file=@path/para/imagem.png"

```

## Desempenho

Com base no script de avaliação, obtivemos o seguinte:
```bash
Confusion Matrix:
[[120   8]
 [  5 123]]

Classification Report:
              precision    recall  f1-score   support
      horses       0.96      0.94      0.95       128
      humans       0.94      0.96      0.95       128
    accuracy                           0.95       256
```
Ou seja, o modelo atende ao problema proposto. A acurácia é alta (95%) e o F1 score está equilibrado entre classes. Temos baixa taxa de falsos positivos/negativos e o threshold customizado otimiza ainda mais a F1.

## Próximos Passos
Fazer análise de threshold para uso na API

Empacotar em container Docker

Testes automatizados de inferência

Implementar data augmentation no treino

Deploy monitorado com Prometheus
