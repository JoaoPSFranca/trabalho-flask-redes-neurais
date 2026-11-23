"""Carregamento e pré-processamento das bases de dados (sem vazamento de dados)."""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ErroValidacaoDados(ValueError):
    """Erro claro para entradas de dados inválidas ou CSV mal formatado."""


@dataclass
class ConjuntoDados:
    """Representa uma base carregada, dividida e pré-processada."""

    nome: str
    chave: str
    tipo: str  # "tabular" | "imagem"
    X_treino: np.ndarray
    X_teste: np.ndarray
    y_treino: np.ndarray
    y_teste: np.ndarray
    X_treino_img: np.ndarray | None
    X_teste_img: np.ndarray | None
    nomes_atributos: list[str]
    nomes_classes: list[str]
    altura_img: int | None = None
    largura_img: int | None = None
    escala_info: str = ""
    total_amostras: int = 0
    subconjunto_info: str = ""


BASES_EMBUTIDAS: dict[str, dict[str, Any]] = {
    "iris": {
        "nome": "Iris (3 classes, 4 atributos)",
        "tipo": "tabular",
        "descricao": "Base clássica de flores Iris (setosa, versicolor, virginica).",
    },
    "breast_cancer": {
        "nome": "Câncer de mama / Breast Cancer (2 classes, 30 atributos)",
        "tipo": "tabular",
        "descricao": "Classificação binária (maligno vs benigno) com atributos clínicos.",
    },
    "digits": {
        "nome": "Dígitos 8x8 / load_digits (10 classes, imagens 8x8)",
        "tipo": "imagem",
        "descricao": "Imagens em escala de cinza 8x8 pixels (0 a 9). Compatível com CNN, MLP e Perceptron.",
    },
    "mnist_amostra": {
        "nome": "MNIST 28x28 — subconjunto de 1.500 imagens (opcional)",
        "tipo": "imagem",
        "descricao": "Subconjunto estratificado do MNIST (28x28). Requer internet apenas no 1º download.",
    },
}


def listar_atributos_embutidos() -> dict[str, list[str]]:
    """Retorna os nomes de atributos das bases embutidas para preencher a interface."""
    iris = load_iris()
    bc = load_breast_cancer()
    digits_attrs = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]
    mnist_attrs = [f"pixel_{r}_{c}" for r in range(28) for c in range(28)]
    return {
        "iris": list(iris.feature_names),
        "breast_cancer": list(bc.feature_names),
        "digits": digits_attrs,
        "mnist_amostra": mnist_attrs,
    }


def _carregar_mnist_amostra(dir_cache: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Carrega ou baixa um subconjunto estratificado de 1.500 imagens do MNIST."""
    if dir_cache is None:
        dir_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance")
    os.makedirs(dir_cache, exist_ok=True)
    caminho_cache = os.path.join(dir_cache, "mnist_amostra_1500.npz")

    if os.path.exists(caminho_cache):
        dados = np.load(caminho_cache, allow_pickle=False)
        X_img = dados["X_img"]
        y = dados["y"]
    else:
        try:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        except Exception as exc:
            raise ErroValidacaoDados(
                "Não foi possível baixar o subconjunto do MNIST (verifique sua conexão com a internet "
                f"ou utilize a base embutida 'Dígitos 8x8'). Detalhe: {exc}"
            ) from exc

        X_full = np.asarray(mnist.data, dtype=np.float32)
        y_full = np.asarray(mnist.target, dtype=int)
        # Subconjunto estratificado de 1.500 imagens para treino rápido na requisição web
        _, X_sub, _, y = train_test_split(
            X_full, y_full, test_size=1500, random_state=42, stratify=y_full
        )
        X_img = (X_sub / 255.0).reshape(-1, 28, 28).astype(np.float32)
        np.savez_compressed(caminho_cache, X_img=X_img, y=y)

    classes = [str(i) for i in range(10)]
    return X_img, y, classes, "Subconjunto estratificado de 1.500 imagens (28x28) escaladas para [0, 1]."


def _decodificar_csv(conteudo_bytes: bytes) -> str:
    """Decodifica bytes de CSV aceitando UTF-8 (com ou sem BOM) e Latin-1."""
    if not conteudo_bytes or not conteudo_bytes.strip():
        raise ErroValidacaoDados("O arquivo CSV enviado está vazio.")
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return conteudo_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ErroValidacaoDados("Não foi possível decodificar o arquivo CSV (use UTF-8 ou Latin-1).")


def _detectar_delimitador(texto: str) -> str:
    """Detecta automaticamente delimitador ',' ou ';' (ou tabulação)."""
    primeiras = [ln for ln in texto.splitlines() if ln.strip()][:5]
    amostra = "\n".join(primeiras)
    if not amostra:
        raise ErroValidacaoDados("O arquivo CSV não contém linhas válidas.")
    contagens = {sep: amostra.count(sep) for sep in (";", ",", "\t")}
    melhor = max(contagens, key=contagens.get)
    return melhor if contagens[melhor] > 0 else ","


def _converter_float(valor: str, sep: str) -> float:
    """Converte string numérica para float, aceitando vírgula decimal quando sep=';'."""
    texto = valor.strip()
    if not texto:
        raise ValueError("valor vazio")
    if sep == ";":
        texto = texto.replace(",", ".")
    numero = float(texto)
    if not np.isfinite(numero):
        raise ValueError("valor não finito")
    return numero


def parse_csv_upload(conteudo_bytes: bytes, nome_arquivo: str = "base_exclusiva.csv") -> dict[str, Any]:
    """Valida e extrai atributos, rótulos e metadados de um CSV tabular (última coluna = classe)."""
    if nome_arquivo and not nome_arquivo.lower().endswith(".csv"):
        raise ErroValidacaoDados(
            f"Formato inválido '{nome_arquivo}'. Envie um arquivo com extensão .csv."
        )

    texto = _decodificar_csv(conteudo_bytes)
    sep = _detectar_delimitador(texto)

    leitor = csv.reader(io.StringIO(texto), delimiter=sep)
    linhas = [ [cel.strip() for cel in linha] for linha in leitor if any(c.strip() for c in linha) ]

    if len(linhas) < 11:
        raise ErroValidacaoDados(
            "O CSV precisa ter cabeçalho e pelo menos 10 registros válidos para divisão treino/teste."
        )

    cabecalho = linhas[0]
    n_cols = len(cabecalho)
    if n_cols < 2:
        raise ErroValidacaoDados(
            "O CSV deve possuir pelo menos 2 colunas (atributos numéricos e última coluna = classe)."
        )
    if any(not col for col in cabecalho):
        raise ErroValidacaoDados("O cabeçalho do CSV contém nomes de coluna vazios.")

    nomes_atributos = cabecalho[:-1]
    linhas_dados = linhas[1:]

    matriz_x: list[list[float]] = []
    alvos_brutos: list[str] = []

    for idx_linha, linha in enumerate(linhas_dados, start=2):
        if len(linha) != n_cols:
            raise ErroValidacaoDados(
                f"CSV mal formatado na linha {idx_linha}: esperadas {n_cols} colunas, encontradas {len(linha)}."
            )
        valores_linha: list[float] = []
        for idx_col, cel in enumerate(linha[:-1]):
            try:
                valores_linha.append(_converter_float(cel, sep))
            except ValueError as exc:
                raise ErroValidacaoDados(
                    f"Valor inválido na linha {idx_linha}, coluna '{nomes_atributos[idx_col]}': "
                    f"'{cel}' (todos os atributos devem ser numéricos e sem valores ausentes)."
                ) from exc

        alvo = linha[-1].strip()
        if not alvo:
            raise ErroValidacaoDados(f"Classe ausente na linha {idx_linha} (última coluna vazia).")
        matriz_x.append(valores_linha)
        alvos_brutos.append(alvo)

    X = np.asarray(matriz_x, dtype=np.float64)

    # Normaliza classes numéricas equivalentes (ex.: "0.0" e "0") se todas forem inteiros
    try:
        alvos_num = [float(a.replace(",", ".")) if sep == ";" else float(a) for a in alvos_brutos]
        if all(np.isfinite(v) and v.is_integer() for v in alvos_num):
            alvos_brutos = [str(int(v)) for v in alvos_num]
    except ValueError:
        pass

    codificador = LabelEncoder()
    y = codificador.fit_transform(alvos_brutos)
    nomes_classes = [str(c) for c in codificador.classes_]

    if len(nomes_classes) < 2:
        raise ErroValidacaoDados(
            f"A última coluna possui apenas 1 classe ('{nomes_classes[0]}'). "
            "A classificação exige pelo menos 2 classes distintas."
        )

    if len(nomes_classes) > 20 and len(nomes_classes) > 0.4 * len(y):
        raise ErroValidacaoDados(
            f"A última coluna '{cabecalho[-1]}' possui {len(nomes_classes)} valores distintos e parece "
            "ser contínua (regressão). A aplicação exige uma base de classificação com classes discretas."
        )

    _, contagens = np.unique(y, return_counts=True)
    if np.min(contagens) < 2:
        classes_raras = [nomes_classes[i] for i, cnt in enumerate(contagens) if cnt < 2]
        raise ErroValidacaoDados(
            "Toda classe precisa ter pelo menos 2 amostras para a divisão estratificada treino/teste. "
            f"Classe(s) com apenas 1 registro: {', '.join(classes_raras)}."
        )

    return {
        "nome": f"CSV exclusivo: {nome_arquivo}",
        "X": X,
        "y": y,
        "nomes_atributos": nomes_atributos,
        "nomes_classes": nomes_classes,
    }


def carregar_e_preparar_dados(
    chave_base: str,
    prop_teste: float,
    semente: int,
    csv_cache: dict[str, Any] | None = None,
) -> ConjuntoDados:
    """Carrega a base escolhida, faz split estratificado e pré-processa sem vazamento de dados."""
    if not (0.05 <= prop_teste <= 0.5):
        raise ErroValidacaoDados("A proporção de teste deve estar entre 5% e 50% (0,05 a 0,50).")

    if chave_base == "iris":
        ds = load_iris()
        X = np.asarray(ds.data, dtype=np.float64)
        y = np.asarray(ds.target, dtype=int)
        nomes_atributos = list(ds.feature_names)
        nomes_classes = [str(c) for c in ds.target_names]
        nome = "Iris"
        tipo = "tabular"
        altura = largura = None
        subconjunto_info = ""

    elif chave_base == "breast_cancer":
        ds = load_breast_cancer()
        X = np.asarray(ds.data, dtype=np.float64)
        y = np.asarray(ds.target, dtype=int)
        nomes_atributos = list(ds.feature_names)
        nomes_classes = [str(c) for c in ds.target_names]
        nome = "Câncer de mama (Breast Cancer)"
        tipo = "tabular"
        altura = largura = None
        subconjunto_info = ""

    elif chave_base == "digits":
        ds = load_digits()
        X_img = np.asarray(ds.images, dtype=np.float32) / 16.0
        X = X_img.reshape(len(X_img), -1).astype(np.float64)
        y = np.asarray(ds.target, dtype=int)
        nomes_atributos = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]
        nomes_classes = [str(c) for c in ds.target_names]
        nome = "Dígitos 8x8 (load_digits)"
        tipo = "imagem"
        altura, largura = 8, 8
        subconjunto_info = "1.797 imagens 8x8 em escala de cinza normalizadas para [0, 1]."

    elif chave_base == "mnist_amostra":
        X_img, y, nomes_classes, subconjunto_info = _carregar_mnist_amostra()
        X = X_img.reshape(len(X_img), -1).astype(np.float64)
        nomes_atributos = [f"pixel_{r}_{c}" for r in range(28) for c in range(28)]
        nome = "MNIST (subconjunto 28x28)"
        tipo = "imagem"
        altura, largura = 28, 28

    elif chave_base == "csv":
        if csv_cache is None:
            raise ErroValidacaoDados(
                "Nenhuma base CSV exclusiva foi carregada. Selecione um arquivo .csv antes de treinar."
            )
        X = np.asarray(csv_cache["X"], dtype=np.float64)
        y = np.asarray(csv_cache["y"], dtype=int)
        nomes_atributos = list(csv_cache["nomes_atributos"])
        nomes_classes = list(csv_cache["nomes_classes"])
        nome = str(csv_cache["nome"])
        tipo = "tabular"
        altura = largura = None
        subconjunto_info = ""

    else:
        raise ErroValidacaoDados(f"Base de dados desconhecida: '{chave_base}'.")

    # Valida se a divisão comporta todas as classes no treino e no teste
    n_classes = len(nomes_classes)
    n_teste = int(round(len(y) * prop_teste))
    n_treino = len(y) - n_teste
    if n_teste < n_classes or n_treino < n_classes:
        raise ErroValidacaoDados(
            f"A proporção de teste ({prop_teste:.0%}) resulta em poucas amostras para representar todas as "
            f"{n_classes} classes no treino ({n_treino}) e no teste ({n_teste}). Ajuste a proporção."
        )

    try:
        X_tr_bruto, X_te_bruto, y_tr, y_te = train_test_split(
            X, y, test_size=prop_teste, random_state=semente, stratify=y
        )
    except ValueError as exc:
        raise ErroValidacaoDados(
            f"Não foi possível dividir a base em treino/teste com estratificação: {exc}"
        ) from exc

    if tipo == "tabular":
        # Padronização ajustada exclusivamente no conjunto de treino (RF5)
        escalonador = StandardScaler()
        X_tr = escalonador.fit_transform(X_tr_bruto)
        X_te = escalonador.transform(X_te_bruto)
        X_tr_img = X_te_img = None
        escala_info = (
            "Atributos tabulares padronizados (z-score) com média e desvio padrão calculados "
            "exclusivamente no conjunto de treino."
        )
    else:
        # Imagens já escaladas para [0, 1]
        X_tr = X_tr_bruto
        X_te = X_te_bruto
        X_tr_img = X_tr.reshape(-1, 1, altura, largura).astype(np.float32)
        X_te_img = X_te.reshape(-1, 1, altura, largura).astype(np.float32)
        escala_info = "Pixels das imagens escalados para o intervalo [0, 1]."

    return ConjuntoDados(
        nome=nome,
        chave=chave_base,
        tipo=tipo,
        X_treino=X_tr,
        X_teste=X_te,
        y_treino=y_tr,
        y_teste=y_te,
        X_treino_img=X_tr_img,
        X_teste_img=X_te_img,
        nomes_atributos=nomes_atributos,
        nomes_classes=nomes_classes,
        altura_img=altura,
        largura_img=largura,
        escala_info=escala_info,
        total_amostras=len(y),
        subconjunto_info=subconjunto_info,
    )
