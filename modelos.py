"""Implementação dos modelos neurais com registro completo de pesos e bias por época."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dados import ConjuntoDados, ErroValidacaoDados


ATIVACOES_PERCEPTRON = ("degrau", "sigmoide", "tanh")
ATIVACOES_MLP = ("sigmoide", "tanh", "relu", "leaky_relu")
ATIVACOES_CNN = ("relu", "leaky_relu", "sigmoide", "tanh")

NOMES_ATIVACOES = {
    "degrau": "Degrau (Heaviside)",
    "sigmoide": "Sigmoide (Logística)",
    "tanh": "Tangente Hiperbólica (tanh)",
    "relu": "ReLU",
    "leaky_relu": "Leaky ReLU",
}


class ErroValidacaoModelo(ValueError):
    """Erro claro para combinações inválidas de modelo, base, ativação ou hiperparâmetros."""


@dataclass
class ResultadoTreino:
    """Resultado padronizado de treinamento, métricas e histórico de parâmetros."""

    modelo: str
    modelo_nome: str
    ativacao: str
    ativacao_nome: str
    arquitetura_resumo: str
    total_parametros: int
    epocas_solicitadas: int
    epocas_executadas: int
    convergiu: bool | None
    tempo_treino_seg: float
    acuracia_treino: float
    acuracia_teste: float
    perda_final: float
    nome_perda: str
    matriz_confusao: np.ndarray
    epocas_eixo: list[int]
    historico_perda: list[float]
    historico_acc_treino: list[float]
    historico_acc_teste: list[float]
    unidade_foco_titulo: str
    nomes_pesos_foco: list[str]
    historico_foco: list[dict[str, Any]]
    tabela_epocas_foco: list[dict[str, Any]]
    tabela_inicial_final: list[dict[str, Any]]
    normas_variacao_camadas: dict[str, list[float]]
    resumo_normas_camadas: list[dict[str, Any]]
    fronteira_perceptron: dict[str, Any] | None = None
    mapas_pesos: dict[str, Any] | None = None
    filtros_cnn: dict[str, Any] | None = None
    observacoes: list[str] = field(default_factory=list)


def _selecionar_epocas_tabela(total_epocas: int) -> list[int]:
    """Seleciona marcos representativos de épocas para a tabela de evolução."""
    if total_epocas <= 6:
        return list(range(total_epocas + 1))
    candidatos = [0, 1, 2, 5, 10, 25, 50, 100, 150, total_epocas]
    selecionadas = sorted({e for e in candidatos if 0 <= e <= total_epocas})
    if len(selecionadas) < 5:
        extras = np.linspace(0, total_epocas, num=5, dtype=int).tolist()
        selecionadas = sorted(set(selecionadas + extras))
    return selecionadas


def _montar_tabelas_foco(
    historico_foco: list[dict[str, Any]],
    nomes_pesos: list[str],
    max_colunas_tabela: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Monta tabela época a época (com Δ total) e tabela completa inicial x final x Δ."""
    ep_final = len(historico_foco) - 1
    epocas_sel = _selecionar_epocas_tabela(ep_final)
    n_cols = min(len(nomes_pesos), max_colunas_tabela)

    tabela_epocas: list[dict[str, Any]] = []
    for ep in epocas_sel:
        reg = historico_foco[ep]
        tabela_epocas.append(
            {
                "epoca": str(ep),
                "eh_delta": False,
                "bias": float(reg["bias"]),
                "pesos": [float(v) for v in reg["pesos"][:n_cols]],
            }
        )

    ini = historico_foco[0]
    fim = historico_foco[-1]
    tabela_epocas.append(
        {
            "epoca": "Δ total",
            "eh_delta": True,
            "bias": float(fim["bias"] - ini["bias"]),
            "pesos": [float(f - i) for i, f in zip(ini["pesos"][:n_cols], fim["pesos"][:n_cols])],
        }
    )

    tabela_inicial_final: list[dict[str, Any]] = [
        {
            "parametro": "bias",
            "nome": "Viés (bias)",
            "inicial": float(ini["bias"]),
            "final": float(fim["bias"]),
            "delta": float(fim["bias"] - ini["bias"]),
        }
    ]
    for idx, nome_attr in enumerate(nomes_pesos):
        v_ini = float(ini["pesos"][idx])
        v_fim = float(fim["pesos"][idx])
        tabela_inicial_final.append(
            {
                "parametro": f"w{idx + 1}",
                "nome": nome_attr,
                "inicial": v_ini,
                "final": v_fim,
                "delta": v_fim - v_ini,
            }
        )

    return tabela_epocas, tabela_inicial_final


def _montar_resumo_normas(
    normas_por_epoca: dict[str, list[float]],
    deltas_totais_camadas: dict[str, float],
) -> list[dict[str, Any]]:
    """Resume a variação total e média de cada camada por norma L2."""
    resumo = []
    for camada, lista in normas_por_epoca.items():
        resumo.append(
            {
                "camada": camada,
                "norma_epoca_1": float(lista[0]) if lista else 0.0,
                "norma_ultima_epoca": float(lista[-1]) if lista else 0.0,
                "norma_media": float(np.mean(lista)) if lista else 0.0,
                "norma_delta_total": float(deltas_totais_camadas[camada]),
            }
        )
    return resumo


class PerceptronNumPy:
    """Perceptron implementado do zero em NumPy (binário e Um-contra-Todos)."""

    def __init__(
        self,
        ativacao: str = "sigmoide",
        taxa_aprendizado: float = 0.1,
        epocas: int = 50,
        tamanho_lote: int = 16,
        semente: int = 42,
    ) -> None:
        if ativacao not in ATIVACOES_PERCEPTRON:
            raise ErroValidacaoModelo(
                f"Ativação '{ativacao}' não é permitida no Perceptron. "
                "Escolha degrau, sigmoide ou tanh (a camada única não usa ReLU/Leaky ReLU neste trabalho)."
            )
        self.ativacao = ativacao
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.semente = semente

    @staticmethod
    def _aplicar_ativacao(z: np.ndarray, nome: str) -> np.ndarray:
        if nome == "degrau":
            return (z >= 0.0).astype(np.float64)
        if nome == "sigmoide":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        if nome == "tanh":
            return np.tanh(np.clip(z, -60.0, 60.0))
        raise ValueError(f"Ativação desconhecida: {nome}")

    def _preparar_alvos(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        if n_classes == 2:
            t = y.reshape(-1, 1).astype(np.float64)
        else:
            t = np.eye(n_classes, dtype=np.float64)[y]
        if self.ativacao == "tanh":
            return 2.0 * t - 1.0
        return t

    def prever_saidas(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        z = X @ W + b
        return self._aplicar_ativacao(z, self.ativacao)

    def prever_classes(self, X: np.ndarray, W: np.ndarray, b: np.ndarray, n_classes: int) -> np.ndarray:
        saidas = self.prever_saidas(X, W, b)
        if n_classes == 2:
            limiar = 0.0 if self.ativacao == "tanh" else 0.5
            return (saidas[:, 0] >= limiar).astype(int)
        return np.argmax(saidas, axis=1)

    def treinar(
        self,
        dados: ConjuntoDados,
        attr_x_idx: int = 0,
        attr_y_idx: int = 1,
    ) -> ResultadoTreino:
        """Treina o Perceptron registrando pesos e bias desde a época 0."""
        t0 = time.perf_counter()
        X_tr, X_te = dados.X_treino, dados.X_teste
        y_tr, y_te = dados.y_treino, dados.y_teste
        n_amostras, n_attrs = X_tr.shape
        n_classes = len(dados.nomes_classes)
        n_saidas = 1 if n_classes == 2 else n_classes

        if n_attrs < 2:
            raise ErroValidacaoModelo(
                "Para desenhar a fronteira de decisão 2D do Perceptron, a base precisa ter pelo menos 2 atributos."
            )
        if not (0 <= attr_x_idx < n_attrs) or not (0 <= attr_y_idx < n_attrs) or attr_x_idx == attr_y_idx:
            raise ErroValidacaoModelo(
                "Selecione dois atributos distintos e válidos para a fronteira de decisão do Perceptron."
            )

        T_tr = self._preparar_alvos(y_tr, n_classes)
        rng = np.random.default_rng(self.semente)

        W = rng.normal(loc=0.0, scale=0.3, size=(n_attrs, n_saidas)).astype(np.float64)
        b = np.zeros(n_saidas, dtype=np.float64)
        W_inicial = W.copy()

        historico_W = [W.copy()]
        historico_b = [b.copy()]

        def avaliar_epoca(W_cur: np.ndarray, b_cur: np.ndarray) -> tuple[float, float, float]:
            saidas_tr = self.prever_saidas(X_tr, W_cur, b_cur)
            perda = float(0.5 * np.mean((T_tr - saidas_tr) ** 2))
            pred_tr = self.prever_classes(X_tr, W_cur, b_cur, n_classes)
            pred_te = self.prever_classes(X_te, W_cur, b_cur, n_classes)
            acc_tr = float(np.mean(pred_tr == y_tr))
            acc_te = float(np.mean(pred_te == y_te))
            return perda, acc_tr, acc_te

        perda0, acc_tr0, acc_te0 = avaliar_epoca(W, b)
        epocas_eixo = [0]
        historico_perda = [perda0]
        historico_acc_tr = [acc_tr0]
        historico_acc_te = [acc_te0]
        normas_camada: list[float] = []

        convergiu: bool | None = False if self.ativacao == "degrau" else None
        lote = min(self.tamanho_lote, n_amostras)

        for epoca in range(1, self.epocas + 1):
            W_ant = W.copy()
            ordem = rng.permutation(n_amostras)

            for ini in range(0, n_amostras, lote):
                idx = ordem[ini : ini + lote]
                X_b = X_tr[idx]
                T_b = T_tr[idx]
                z_b = X_b @ W + b
                y_b = self._aplicar_ativacao(z_b, self.ativacao)

                if self.ativacao == "degrau":
                    delta = T_b - y_b
                elif self.ativacao == "sigmoide":
                    delta = (T_b - y_b) * (y_b * (1.0 - y_b))
                else:  # tanh
                    delta = (T_b - y_b) * (1.0 - y_b ** 2)

                W = W + self.taxa_aprendizado * ((X_b.T @ delta) / len(X_b))
                b = b + self.taxa_aprendizado * np.mean(delta, axis=0)

            perda_ep, acc_tr_ep, acc_te_ep = avaliar_epoca(W, b)
            historico_W.append(W.copy())
            historico_b.append(b.copy())
            epocas_eixo.append(epoca)
            historico_perda.append(perda_ep)
            historico_acc_tr.append(acc_tr_ep)
            historico_acc_te.append(acc_te_ep)
            normas_camada.append(float(np.linalg.norm(W - W_ant)))

            if self.ativacao == "degrau":
                saidas_bin = self.prever_saidas(X_tr, W, b)
                if np.array_equal(saidas_bin, T_tr):
                    convergiu = True
                    break

        tempo = time.perf_counter() - t0
        pred_te_final = self.prever_classes(X_te, W, b, n_classes)
        cm = confusion_matrix(y_te, pred_te_final, labels=list(range(n_classes)))

        # Histórico do neurônio 1 (índice 0 da camada de saída)
        historico_foco = [
            {"epoca": ep, "bias": float(historico_b[ep][0]), "pesos": historico_W[ep][:, 0].copy()}
            for ep in range(len(historico_W))
        ]
        tabela_epocas, tabela_inicial_final = _montar_tabelas_foco(historico_foco, dados.nomes_atributos)

        nome_camada = "Camada de Saída (Perceptron)"
        normas_dict = {nome_camada: normas_camada}
        deltas_totais = {nome_camada: float(np.linalg.norm(W - W_inicial))}
        resumo_normas = _montar_resumo_normas(normas_dict, deltas_totais)

        ep_final = len(historico_W) - 1
        epocas_fronteira = sorted({e for e in (0, 1, 5, ep_final) if 0 <= e <= ep_final})
        if len(epocas_fronteira) < 4 and ep_final >= 3:
            epocas_fronteira = sorted(set(np.linspace(0, ep_final, num=4, dtype=int).tolist()))

        retas_fronteira = []
        for ep in epocas_fronteira:
            w_ep = historico_W[ep][:, 0]
            b_ep = float(historico_b[ep][0])
            retas_fronteira.append(
                {
                    "epoca": ep,
                    "w_x": float(w_ep[attr_x_idx]),
                    "w_y": float(w_ep[attr_y_idx]),
                    "bias": b_ep,
                    "acc_treino": historico_acc_tr[ep],
                }
            )

        if n_classes == 2:
            foco_titulo = f"Neurônio 1 (saída binária: {dados.nomes_classes[1]} vs {dados.nomes_classes[0]})"
            subtitulo_fronteira = (
                f"Reta do neurônio de saída ({dados.nomes_classes[1]} vs {dados.nomes_classes[0]}); "
                "demais atributos na média padronizada (= 0)."
            )
        else:
            foco_titulo = f"Neurônio 1 (saída OvA — classe '{dados.nomes_classes[0]}' vs demais)"
            subtitulo_fronteira = (
                f"Reta do neurônio 1 ('{dados.nomes_classes[0]}' vs demais classes, Um-contra-Todos); "
                "demais atributos na média padronizada (= 0)."
            )

        labels_saidas = (
            [f"Saída ({dados.nomes_classes[1]})"]
            if n_classes == 2
            else [f"Neurônio {i + 1} ({cls})" for i, cls in enumerate(dados.nomes_classes)]
        )
        mapas_pesos = {
            "titulo": "Pesos do Perceptron — Inicial (Época 0) vs Final",
            "W_inicial": W_inicial.T,  # (n_saidas, n_attrs)
            "W_final": W.T,
            "labels_x": dados.nomes_atributos,
            "labels_y": labels_saidas,
        }

        observacoes = []
        if self.ativacao == "degrau":
            if convergiu:
                observacoes.append(
                    f"O Perceptron com degrau atingiu 0 erros no conjunto de treino na época {ep_final} e encerrou."
                )
            else:
                observacoes.append(
                    "O Perceptron com degrau não zerou os erros no limite de épocas (classes não linearmente separáveis ou taxa/épocas insuficientes)."
                )
        else:
            observacoes.append(
                f"Treinado com a regra delta (gradiente do erro quadrático com derivada da {self.ativacao})."
            )
        if n_classes > 2:
            observacoes.append(
                f"Problema multiclasse ({n_classes} classes): implementada estratégia Um-contra-Todos com {n_classes} neurônios de saída."
            )

        return ResultadoTreino(
            modelo="perceptron",
            modelo_nome="Perceptron (NumPy)",
            ativacao=self.ativacao,
            ativacao_nome=NOMES_ATIVACOES[self.ativacao],
            arquitetura_resumo=(
                f"{n_attrs} entradas → {n_saidas} neurônio(s) de saída "
                f"({'binário' if n_classes == 2 else 'Um-contra-Todos'})"
            ),
            total_parametros=int(W.size + b.size),
            epocas_solicitadas=self.epocas,
            epocas_executadas=ep_final,
            convergiu=convergiu,
            tempo_treino_seg=tempo,
            acuracia_treino=historico_acc_tr[-1],
            acuracia_teste=historico_acc_te[-1],
            perda_final=historico_perda[-1],
            nome_perda="Perda (EQM / 2)",
            matriz_confusao=cm,
            epocas_eixo=epocas_eixo,
            historico_perda=historico_perda,
            historico_acc_treino=historico_acc_tr,
            historico_acc_teste=historico_acc_te,
            unidade_foco_titulo=foco_titulo,
            nomes_pesos_foco=dados.nomes_atributos,
            historico_foco=historico_foco,
            tabela_epocas_foco=tabela_epocas,
            tabela_inicial_final=tabela_inicial_final,
            normas_variacao_camadas=normas_dict,
            resumo_normas_camadas=resumo_normas,
            fronteira_perceptron={
                "attr_x_idx": attr_x_idx,
                "attr_y_idx": attr_y_idx,
                "attr_x_nome": dados.nomes_atributos[attr_x_idx],
                "attr_y_nome": dados.nomes_atributos[attr_y_idx],
                "retas": retas_fronteira,
                "subtitulo": subtitulo_fronteira,
            },
            mapas_pesos=mapas_pesos,
            observacoes=observacoes,
        )


def _criar_ativacao_torch(nome: str, modelo_nome: str) -> nn.Module:
    """Cria a camada de ativação PyTorch validando compatibilidade com treino por gradiente."""
    if nome == "degrau":
        raise ErroValidacaoModelo(
            f"A função 'degrau' não pode ser usada no {modelo_nome}: sua derivada é nula "
            "(f'(z) = 0 para z ≠ 0), o que impede o aprendizado por gradiente/retropropagação."
        )
    mapa = {
        "sigmoide": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    }
    if nome not in mapa:
        raise ErroValidacaoModelo(f"Função de ativação desconhecida: '{nome}'.")
    return mapa[nome]()


def _parse_neuronios_mlp(n_camadas: int, spec_neuronios: str) -> list[int]:
    """Interpreta a configuração de neurônios das camadas ocultas (ex.: '16' ou '16, 8')."""
    if not (1 <= n_camadas <= 5):
        raise ErroValidacaoModelo("O número de camadas ocultas do MLP deve estar entre 1 e 5.")

    partes = [p.strip() for p in str(spec_neuronios).split(",") if p.strip()]
    if not partes:
        raise ErroValidacaoModelo("Informe o número de neurônios por camada oculta do MLP.")

    try:
        valores = [int(p) for p in partes]
    except ValueError as exc:
        raise ErroValidacaoModelo(
            "Neurônios por camada devem ser números inteiros (ex.: '16' ou '16, 8')."
        ) from exc

    if any(v < 1 or v > 256 for v in valores):
        raise ErroValidacaoModelo("Cada camada oculta do MLP deve ter entre 1 e 256 neurônios.")

    if len(valores) == 1:
        return valores * n_camadas
    if len(valores) == n_camadas:
        return valores
    raise ErroValidacaoModelo(
        f"Você definiu {n_camadas} camadas ocultas, mas informou {len(valores)} valores ({partes}). "
        f"Informe apenas 1 valor (repetido em todas) ou exatamente {n_camadas} valores separados por vírgula."
    )


class MLPPyTorch:
    """Perceptron Multicamadas (MLP) em PyTorch com histórico completo de parâmetros por época."""

    def __init__(
        self,
        ativacao: str = "relu",
        taxa_aprendizado: float = 0.1,
        epocas: int = 50,
        tamanho_lote: int = 16,
        n_camadas_ocultas: int = 2,
        neuronios_ocultos: str = "16, 8",
        semente: int = 42,
    ) -> None:
        if ativacao not in ATIVACOES_MLP:
            _criar_ativacao_torch(ativacao, "MLP")
            raise ErroValidacaoModelo(f"Ativação '{ativacao}' não permitida para MLP.")
        self.ativacao = ativacao
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.arquitetura_oculta = _parse_neuronios_mlp(n_camadas_ocultas, neuronios_ocultos)
        self.semente = semente

    def treinar(self, dados: ConjuntoDados) -> ResultadoTreino:
        """Treina o MLP com SGD, registrando pesos/bias da época 0 até a época final."""
        t0 = time.perf_counter()
        torch.manual_seed(self.semente)
        np.random.seed(self.semente)

        X_tr = torch.tensor(dados.X_treino, dtype=torch.float32)
        X_te = torch.tensor(dados.X_teste, dtype=torch.float32)
        n_classes = len(dados.nomes_classes)
        n_attrs = dados.X_treino.shape[1]
        binario = n_classes == 2

        if binario:
            y_tr_t = torch.tensor(dados.y_treino.reshape(-1, 1), dtype=torch.float32)
            criterio = nn.BCEWithLogitsLoss()
            n_saidas = 1
        else:
            y_tr_t = torch.tensor(dados.y_treino, dtype=torch.long)
            criterio = nn.CrossEntropyLoss()
            n_saidas = n_classes

        camadas_seq: list[nn.Module] = []
        camadas_lineares: list[tuple[str, nn.Linear]] = []
        dim_atual = n_attrs
        for idx, h in enumerate(self.arquitetura_oculta, start=1):
            lin = nn.Linear(dim_atual, h)
            camadas_lineares.append((f"Oculta {idx} ({dim_atual}→{h})", lin))
            camadas_seq.append(lin)
            camadas_seq.append(_criar_ativacao_torch(self.ativacao, "MLP"))
            dim_atual = h

        camada_saida = nn.Linear(dim_atual, n_saidas)
        camadas_lineares.append((f"Saída ({dim_atual}→{n_saidas})", camada_saida))
        camadas_seq.append(camada_saida)

        rede = nn.Sequential(*camadas_seq)
        otimizador = torch.optim.SGD(rede.parameters(), lr=self.taxa_aprendizado)

        pesos_iniciais = {
            nome: layer.weight.detach().cpu().numpy().copy() for nome, layer in camadas_lineares
        }

        def avaliar() -> tuple[float, float, float, np.ndarray]:
            rede.eval()
            with torch.no_grad():
                logits_tr = rede(X_tr)
                perda = float(criterio(logits_tr, y_tr_t).item())
                logits_te = rede(X_te)
                if binario:
                    pred_tr = (torch.sigmoid(logits_tr).squeeze(1) >= 0.5).cpu().numpy().astype(int)
                    pred_te = (torch.sigmoid(logits_te).squeeze(1) >= 0.5).cpu().numpy().astype(int)
                else:
                    pred_tr = torch.argmax(logits_tr, dim=1).cpu().numpy()
                    pred_te = torch.argmax(logits_te, dim=1).cpu().numpy()
                acc_tr = float(np.mean(pred_tr == dados.y_treino))
                acc_te = float(np.mean(pred_te == dados.y_teste))
            return perda, acc_tr, acc_te, pred_te

        perda0, acc_tr0, acc_te0, pred_te_final = avaliar()
        epocas_eixo = [0]
        historico_perda = [perda0]
        historico_acc_tr = [acc_tr0]
        historico_acc_te = [acc_te0]

        primeira_camada = camadas_lineares[0][1]
        historico_foco = [
            {
                "epoca": 0,
                "bias": float(primeira_camada.bias[0].item()),
                "pesos": primeira_camada.weight[0].detach().cpu().numpy().copy(),
            }
        ]
        normas_camadas: dict[str, list[float]] = {nome: [] for nome, _ in camadas_lineares}

        dataset_tr = TensorDataset(X_tr, y_tr_t)
        lote = min(self.tamanho_lote, len(X_tr))
        divergiu = False
        epoca_divergencia = None

        for epoca in range(1, self.epocas + 1):
            pesos_antes = {
                nome: layer.weight.detach().cpu().numpy().copy() for nome, layer in camadas_lineares
            }
            gerador = torch.Generator().manual_seed(self.semente + epoca)
            loader = DataLoader(dataset_tr, batch_size=lote, shuffle=True, generator=gerador)

            rede.train()
            for xb, yb in loader:
                otimizador.zero_grad()
                saida_b = rede(xb)
                loss_b = criterio(saida_b, yb)
                loss_b.backward()
                otimizador.step()

            perda_ep, acc_tr_ep, acc_te_ep, pred_te_ep = avaliar()
            if not np.isfinite(perda_ep):
                divergiu = True
                epoca_divergencia = epoca
                break

            pred_te_final = pred_te_ep
            epocas_eixo.append(epoca)
            historico_perda.append(perda_ep)
            historico_acc_tr.append(acc_tr_ep)
            historico_acc_te.append(acc_te_ep)

            for nome, layer in camadas_lineares:
                w_atual = layer.weight.detach().cpu().numpy().copy()
                normas_camadas[nome].append(float(np.linalg.norm(w_atual - pesos_antes[nome])))

            historico_foco.append(
                {
                    "epoca": epoca,
                    "bias": float(primeira_camada.bias[0].item()),
                    "pesos": primeira_camada.weight[0].detach().cpu().numpy().copy(),
                }
            )

        tempo = time.perf_counter() - t0
        cm = confusion_matrix(dados.y_teste, pred_te_final, labels=list(range(n_classes)))
        tabela_epocas, tabela_inicial_final = _montar_tabelas_foco(historico_foco, dados.nomes_atributos)

        deltas_totais = {
            nome: float(np.linalg.norm(layer.weight.detach().cpu().numpy() - pesos_iniciais[nome]))
            for nome, layer in camadas_lineares
        }
        resumo_normas = _montar_resumo_normas(normas_camadas, deltas_totais)

        nome_oculta1, layer_oculta1 = camadas_lineares[0]
        mapas_pesos = {
            "titulo": f"Mapa de calor dos pesos — {nome_oculta1} (Inicial x Final)",
            "W_inicial": pesos_iniciais[nome_oculta1],
            "W_final": layer_oculta1.weight.detach().cpu().numpy().copy(),
            "labels_x": dados.nomes_atributos,
            "labels_y": [f"Oculto {i + 1}" for i in range(layer_oculta1.out_features)],
        }

        total_params = sum(p.numel() for p in rede.parameters())
        arq_str = " → ".join([str(n_attrs)] + [str(h) for h in self.arquitetura_oculta] + [str(n_saidas)])
        saida_desc = "1 neurônio (sigmoide)" if binario else f"{n_saidas} neurônios (softmax)"

        observacoes = [
            f"MLP treinado com SGD (η={self.taxa_aprendizado}) e saída de {saida_desc}.",
            f"O gráfico de ajuste acompanha o Neurônio 1 da 1ª camada oculta ({self.arquitetura_oculta[0]} neurônios).",
        ]
        if divergiu:
            observacoes.append(
                f"Atenção: ocorreu instabilidade numérica na época {epoca_divergencia} (taxa de aprendizado muito elevada)."
            )

        return ResultadoTreino(
            modelo="mlp",
            modelo_nome="MLP (PyTorch)",
            ativacao=self.ativacao,
            ativacao_nome=NOMES_ATIVACOES[self.ativacao],
            arquitetura_resumo=f"Rede {arq_str}",
            total_parametros=int(total_params),
            epocas_solicitadas=self.epocas,
            epocas_executadas=len(epocas_eixo) - 1,
            convergiu=None,
            tempo_treino_seg=tempo,
            acuracia_treino=historico_acc_tr[-1],
            acuracia_teste=historico_acc_te[-1],
            perda_final=historico_perda[-1],
            nome_perda="Entropia Cruzada" + (" Binária" if binario else ""),
            matriz_confusao=cm,
            epocas_eixo=epocas_eixo,
            historico_perda=historico_perda,
            historico_acc_treino=historico_acc_tr,
            historico_acc_teste=historico_acc_te,
            unidade_foco_titulo="Neurônio 1 (1ª camada oculta)",
            nomes_pesos_foco=dados.nomes_atributos,
            historico_foco=historico_foco,
            tabela_epocas_foco=tabela_epocas,
            tabela_inicial_final=tabela_inicial_final,
            normas_variacao_camadas=normas_camadas,
            resumo_normas_camadas=resumo_normas,
            mapas_pesos=mapas_pesos,
            observacoes=observacoes,
        )


def executar_treinamento(dados: ConjuntoDados, config: dict[str, Any]) -> ResultadoTreino:
    """Valida compatibilidade modelo/base/ativação/hiperparâmetros e executa o treino."""
    modelo = config["modelo"]
    ativacao = config["ativacao"]
    taxa = float(config["taxa_aprendizado"])
    epocas = int(config["epocas"])
    lote = int(config["tamanho_lote"])
    semente = int(config["semente"])

    if taxa <= 0 or taxa > 10:
        raise ErroValidacaoModelo("A taxa de aprendizado deve ser maior que 0 e no máximo 10.")
    if lote < 1:
        raise ErroValidacaoModelo("O tamanho do lote deve ser pelo menos 1.")

    if modelo == "perceptron":
        if not (1 <= epocas <= 200):
            raise ErroValidacaoModelo("Para o Perceptron, escolha entre 1 e 200 épocas.")
        perceptron = PerceptronNumPy(
            ativacao=ativacao,
            taxa_aprendizado=taxa,
            epocas=epocas,
            tamanho_lote=lote,
            semente=semente,
        )
        return perceptron.treinar(
            dados,
            attr_x_idx=int(config.get("attr_x", 0)),
            attr_y_idx=int(config.get("attr_y", 1)),
        )

    if modelo == "mlp":
        if not (1 <= epocas <= 200):
            raise ErroValidacaoModelo("Para o MLP, escolha entre 1 e 200 épocas.")
        mlp = MLPPyTorch(
            ativacao=ativacao,
            taxa_aprendizado=taxa,
            epocas=epocas,
            tamanho_lote=lote,
            n_camadas_ocultas=int(config.get("n_camadas_ocultas", 2)),
            neuronios_ocultos=str(config.get("neuronios_ocultos", "16, 8")),
            semente=semente,
        )
        return mlp.treinar(dados)

    if modelo == "cnn":
        raise ErroValidacaoModelo(
            "O modelo 'CNN' será habilitado na Etapa 3."
        )

    raise ErroValidacaoModelo(f"Modelo inválido: '{modelo}'.")
