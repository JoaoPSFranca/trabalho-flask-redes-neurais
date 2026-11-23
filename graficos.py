"""Geração de gráficos Matplotlib convertidos em base64 PNG para exibição no Flask."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dados import ConjuntoDados
from modelos import ResultadoTreino

CORES_CURVAS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#17BECF",
]


def _fig_para_base64(fig: plt.Figure) -> str:
    """Converte uma figura Matplotlib em string PNG base64 e libera memória."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=135, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    finally:
        plt.close(fig)


def grafico_perda_acuracia(resultado: ResultadoTreino) -> str:
    """Gráfico lado a lado com perda e acurácia (treino e teste) por época."""
    epocas = resultado.epocas_eixo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))

    ax1.plot(epocas, resultado.historico_perda, color="#C0392B", lw=2.1, marker="o" if len(epocas) <= 20 else None, markersize=4)
    ax1.set_title("Perda ao longo das épocas", fontsize=11.5, fontweight="bold", color="#1F3864")
    ax1.set_xlabel("Época (0 = inicial)")
    ax1.set_ylabel(resultado.nome_perda)
    ax1.grid(True, linestyle="--", alpha=0.35)

    acc_tr_pct = [a * 100.0 for a in resultado.historico_acc_treino]
    acc_te_pct = [a * 100.0 for a in resultado.historico_acc_teste]
    marcador = "o" if len(epocas) <= 20 else None
    ax2.plot(epocas, acc_tr_pct, color="#1F3864", lw=2.1, marker=marcador, markersize=4, label="Treino")
    ax2.plot(epocas, acc_te_pct, color="#2E8B57", lw=2.1, linestyle="--", marker=marcador, markersize=4, label="Teste")
    ax2.set_title("Acurácia por época", fontsize=11.5, fontweight="bold", color="#1F3864")
    ax2.set_xlabel("Época (0 = inicial)")
    ax2.set_ylabel("Acurácia (%)")
    ax2.set_ylim(-2, 104)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    return _fig_para_base64(fig)


def grafico_matriz_confusao(cm: np.ndarray, nomes_classes: list[str]) -> str:
    """Matriz de confusão do conjunto de teste com valores anotados."""
    n = len(nomes_classes)
    lado = max(4.4, min(7.8, 3.4 + 0.45 * n))
    fig, ax = plt.subplots(figsize=(lado + 0.6, lado))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    rotulos = [c if len(c) <= 14 else c[:12] + "…" for c in nomes_classes]
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=rotulos,
        yticklabels=rotulos,
        xlabel="Classe prevista",
        ylabel="Classe real",
        title="Matriz de confusão (conjunto de teste)",
    )
    ax.title.set_color("#1F3864")
    ax.title.set_fontweight("bold")
    if n > 4:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    limiar = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > limiar else "#1E293B",
                fontsize=9 if n >= 8 else 11,
                fontweight="bold",
            )

    fig.tight_layout()
    return _fig_para_base64(fig)


def grafico_evolucao_parametros(resultado: ResultadoTreino) -> str:
    """Gráfico de linhas com a trajetória dos primeiros pesos e do bias ao longo das épocas."""
    hist = resultado.historico_foco
    epocas = [r["epoca"] for r in hist]
    biases = [r["bias"] for r in hist]
    matriz_pesos = np.array([r["pesos"] for r in hist])  # (E+1, d)

    max_curvas = min(8, matriz_pesos.shape[1])
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    marcador = "o" if len(epocas) <= 18 else None

    for j in range(max_curvas):
        nome = resultado.nomes_pesos_foco[j]
        rotulo = f"w[{nome}]" if len(nome) <= 22 else f"w{j + 1} [{nome[:19]}…]"
        ax.plot(
            epocas,
            matriz_pesos[:, j],
            color=CORES_CURVAS[j % len(CORES_CURVAS)],
            lw=2.0,
            marker=marcador,
            markersize=4,
            label=rotulo,
        )

    ax.plot(
        epocas,
        biases,
        color="#111827",
        lw=2.2,
        linestyle="--",
        marker=marcador,
        markersize=4,
        label="bias",
    )

    ax.axhline(0.0, color="#94A3B8", lw=0.8, linestyle=":")
    ax.set_title(f"Ajuste dos parâmetros — {resultado.unidade_foco_titulo}", fontsize=11.5, fontweight="bold", color="#1F3864")
    ax.set_xlabel("Época (0 = valores iniciais)")
    ax.set_ylabel("Valor do parâmetro")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2 if max_curvas > 3 else 1, fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    return _fig_para_base64(fig)


def grafico_norma_camadas(resultado: ResultadoTreino) -> str:
    """Gráfico da norma L2 da variação dos pesos por época em cada camada."""
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    for idx, (camada, valores) in enumerate(resultado.normas_variacao_camadas.items()):
        epocas = list(range(1, len(valores) + 1))
        marcador = "o" if len(epocas) <= 20 else None
        ax.plot(
            epocas,
            valores,
            color=CORES_CURVAS[idx % len(CORES_CURVAS)],
            lw=2.1,
            marker=marcador,
            markersize=4,
            label=camada,
        )

    ax.set_title("Norma da variação dos pesos por época (||W(t) − W(t−1)||₂)", fontsize=11.5, fontweight="bold", color="#1F3864")
    ax.set_xlabel("Época")
    ax.set_ylabel("Norma L2 da variação")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    return _fig_para_base64(fig)


def grafico_fronteira_perceptron(dados: ConjuntoDados, resultado: ResultadoTreino) -> str | None:
    """Desenha a reta da fronteira de decisão do Perceptron em épocas selecionadas sobre o treino."""
    info = resultado.fronteira_perceptron
    if not info:
        return None

    ix = info["attr_x_idx"]
    iy = info["attr_y_idx"]
    retas = info["retas"]
    n_paineis = len(retas)

    X_tr = dados.X_treino
    y_tr = dados.y_treino
    x_vals = X_tr[:, ix]
    y_vals = X_tr[:, iy]
    pad_x = max(0.35, 0.1 * float(x_vals.max() - x_vals.min()))
    pad_y = max(0.35, 0.1 * float(y_vals.max() - y_vals.min()))
    xlim = (float(x_vals.min()) - pad_x, float(x_vals.max()) + pad_x)
    ylim = (float(y_vals.min()) - pad_y, float(y_vals.max()) + pad_y)
    xs_grid = np.linspace(xlim[0], xlim[1], 250)

    fig, axes = plt.subplots(1, n_paineis, figsize=(4.1 * n_paineis, 4.1), sharex=True, sharey=True)
    if n_paineis == 1:
        axes = [axes]

    marcadores = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]
    n_classes = len(dados.nomes_classes)

    for idx_ax, (ax, reta) in enumerate(zip(axes, retas)):
        for c in range(n_classes):
            mascara = y_tr == c
            if not np.any(mascara):
                continue
            rotulo = dados.nomes_classes[c] if idx_ax == 0 and n_classes <= 6 else None
            ax.scatter(
                x_vals[mascara],
                y_vals[mascara],
                s=24,
                alpha=0.78,
                color=CORES_CURVAS[c % len(CORES_CURVAS)],
                marker=marcadores[c % len(marcadores)],
                edgecolors="white",
                linewidths=0.4,
                label=rotulo,
            )

        wx, wy, b = reta["w_x"], reta["w_y"], reta["bias"]
        if abs(wy) > 1e-8:
            ys_linha = -(wx * xs_grid + b) / wy
            ax.plot(xs_grid, ys_linha, color="#111827", lw=2.1, linestyle="--", label="Fronteira" if idx_ax == 0 else None)
        elif abs(wx) > 1e-8:
            x_vert = -b / wx
            ax.axvline(x_vert, color="#111827", lw=2.1, linestyle="--", label="Fronteira" if idx_ax == 0 else None)
        else:
            ax.text(
                0.5,
                0.5,
                "wx = wy = 0\n(sem reta definida)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"{info['attr_x_nome']} (padron.)")
        if idx_ax == 0:
            ax.set_ylabel(f"{info['attr_y_nome']} (padron.)")
        titulo_ep = "Inicial (época 0)" if reta["epoca"] == 0 else f"Época {reta['epoca']}"
        ax.set_title(f"{titulo_ep}\nAcur. treino: {reta['acc_treino']:.1%}", fontsize=10.5, fontweight="bold", color="#1F3864")
        ax.grid(True, linestyle="--", alpha=0.3)

    if n_classes <= 6:
        axes[0].legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9, loc="best")

    fig.suptitle(info["subtitulo"], fontsize=10.5, y=1.03, color="#334155")
    fig.tight_layout()
    return _fig_para_base64(fig)


def grafico_mapa_pesos(mapas_pesos: dict[str, Any] | None) -> str | None:
    """Compara mapa de calor dos pesos iniciais, finais e variação (Δ)."""
    if not mapas_pesos:
        return None

    W_ini = np.asarray(mapas_pesos["W_inicial"], dtype=float)
    W_fim = np.asarray(mapas_pesos["W_final"], dtype=float)
    W_delta = W_fim - W_ini

    lim = max(float(np.max(np.abs(W_ini))), float(np.max(np.abs(W_fim))), 1e-6)
    lim_d = max(float(np.max(np.abs(W_delta))), 1e-6)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, max(3.6, min(6.0, 2.2 + 0.35 * W_ini.shape[0]))))

    for ax, mat, titulo, vlim, cmap in (
        (ax1, W_ini, "Inicial (Época 0)", lim, "RdBu_r"),
        (ax2, W_fim, "Final", lim, "RdBu_r"),
        (ax3, W_delta, "Variação (Δ = Final − Inicial)", lim_d, "PiYG"),
    ):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim)
        ax.set_title(titulo, fontsize=10.5, fontweight="bold", color="#1F3864")
        ax.set_xlabel("Entradas / Atributos")
        if ax is ax1:
            ax.set_ylabel("Neurônios")
        if mat.shape[0] <= 10 and mat.shape[1] <= 10:
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    ax.text(
                        c,
                        r,
                        f"{mat[r, c]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if abs(mat[r, c]) > 0.6 * vlim else "#111827",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(mapas_pesos.get("titulo", "Mapa de pesos inicial x final"), fontsize=11.5, fontweight="bold", y=1.03, color="#1F3864")
    fig.tight_layout()
    return _fig_para_base64(fig)


def montar_todos_graficos(dados: ConjuntoDados, resultado: ResultadoTreino) -> dict[str, str | None]:
    """Gera todos os gráficos pertinentes ao modelo treinado."""
    return {
        "perda_acuracia": grafico_perda_acuracia(resultado),
        "matriz_confusao": grafico_matriz_confusao(resultado.matriz_confusao, dados.nomes_classes),
        "evolucao_parametros": grafico_evolucao_parametros(resultado),
        "norma_camadas": grafico_norma_camadas(resultado),
        "fronteira_perceptron": grafico_fronteira_perceptron(dados, resultado),
        "mapa_pesos": grafico_mapa_pesos(resultado.mapas_pesos),
    }
