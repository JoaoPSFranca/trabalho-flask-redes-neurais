# Aplicação Flask — Perceptron, MLP e CNN

Aplicação web desenvolvida em Flask para treinar e inspecionar o funcionamento interno de três arquiteturas de redes neurais (**Perceptron**, **MLP** e **CNN**), com foco no acompanhamento época a época dos pesos, bias, normas de atualização, fronteiras de decisão e filtros convolucionais.

---

## 📁 Estrutura do Projeto

```text
trabalho_flask/
├── app.py                 # Rotas Flask (GET / e POST /), cache de CSV e validação de formulário
├── modelos.py             # Perceptron (NumPy do zero), MLP (PyTorch) e CNN (PyTorch)
├── dados.py               # Carregamento de bases embutidas, upload CSV e pré-processamento sem vazamento
├── graficos.py            # Geração de gráficos Matplotlib em base64 PNG
├── requirements.txt       # Dependências do projeto
├── templates/
│   └── index.html         # Interface web responsiva com painel de resultados e indicador de treino
├── tests/
│   └── test_app.py        # Suíte de testes automatizados (RF1 a RF12)
└── README.md              # Instruções de instalação e uso
```

---

## ⚙️ Instalação e Execução

1. Acesse a pasta do projeto:

```bash
cd atividades/atividade_redes_neurais/trabalho_flask
```

2. (Opcional, recomendado) Crie e ative um ambiente virtual Python:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Inicie a aplicação Flask:

```bash
python app.py
```

5. Abra no navegador o endereço exibido no terminal (por padrão `http://127.0.0.1:5000`).

---

## 🧠 Modelos e Ativações Suportados

| Modelo | Implementação | Ativações Permitidas | Observações |
| :--- | :--- | :--- | :--- |
| **Perceptron** | NumPy do zero | `degrau`, `sigmoide`, `tanh` | Binário (1 neurônio) ou multiclasse (Um-contra-Todos). Regra clássica com `degrau` e regra delta com `sigmoide`/`tanh`. |
| **MLP** | PyTorch (`SGD`) | `sigmoide`, `tanh`, `relu`, `leaky_relu` | Número de camadas ocultas (1 a 5) e neurônios por camada configuráveis (ex.: `16` ou `16, 8`). Rejeita `degrau` por ter derivada nula. |
| **CNN** | PyTorch (`SGD`) | `relu`, `leaky_relu`, `sigmoide`, `tanh` | `Conv2D → Ativação → MaxPool2D(2×2) → Flatten → Densa (Softmax)`. Exige base de imagens e até 30 épocas. |

---

## 📊 Bases de Dados e CSV Exclusivo

- **Iris**: 150 amostras, 4 atributos tabulares, 3 classes.
- **Câncer de mama (`Breast Cancer`)**: 569 amostras, 30 atributos tabulares, 2 classes.
- **Dígitos 8×8 (`load_digits`)**: 1.797 imagens 8×8 em escala de cinza, 10 classes (compatível com CNN, MLP e Perceptron).
- **MNIST 28×28 (opcional)**: subconjunto estratificado de 1.500 imagens baixado sob demanda e salvo em cache local.
- **CSV exclusivo do aluno**: upload direto na interface (aceita delimitador `,` ou `;`, com última coluna = classe e atributos numéricos). A troca de base ou modelo não exige reiniciar o servidor.

### Prevenção de Vazamento de Dados

- A padronização (`StandardScaler`) de atributos tabulares ajusta média e desvio padrão **exclusivamente no conjunto de treino**, aplicando a mesma transformação ao conjunto de teste.
- As imagens são escaladas para o intervalo `[0, 1]`.

---

## 🧪 Como Executar os Testes Automatizados

Na pasta `trabalho_flask/`:

```bash
python -m unittest discover -s tests -v
```
