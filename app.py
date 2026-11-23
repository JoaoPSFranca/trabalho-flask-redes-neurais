"""Aplicação web Flask para treinamento e análise de redes neurais."""

from __future__ import annotations

import json
import os
import uuid
from collections import OrderedDict
from typing import Any

from flask import Flask, render_template, request

from dados import (
    BASES_EMBUTIDAS,
    ErroValidacaoDados,
    carregar_e_preparar_dados,
    listar_atributos_embutidos,
    parse_csv_upload,
)
from graficos import montar_todos_graficos
from modelos import (
    NOMES_ATIVACOES,
    ErroValidacaoModelo,
    executar_treinamento,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Cache em memória para manter a base CSV enviada entre requisições sem reiniciar o servidor (RF2)
_CSV_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_CACHE_CSV = 8
_ATRIBUTOS_EMBUTIDOS = listar_atributos_embutidos()


def _guardar_csv_cache(dados_csv: dict[str, Any]) -> str:
    token = uuid.uuid4().hex[:12]
    _CSV_CACHE[token] = dados_csv
    while len(_CSV_CACHE) > _MAX_CACHE_CSV:
        _CSV_CACHE.popitem(last=False)
    return token


def _obter_csv_cache(token: str | None) -> dict[str, Any] | None:
    if token and token in _CSV_CACHE:
        return _CSV_CACHE[token]
    if _CSV_CACHE:
        ultima_chave = next(reversed(_CSV_CACHE))
        return _CSV_CACHE[ultima_chave]
    return None


def _estado_inicial_formulario() -> dict[str, Any]:
    return {
        "modelo": "perceptron",
        "base": "iris",
        "ativacao": "sigmoide",
        "taxa_aprendizado": "0.1",
        "epocas": "50",
        "tamanho_lote": "16",
        "prop_teste": "25",
        "semente": "42",
        "attr_x": "0",
        "attr_y": "1",
        "n_camadas_ocultas": "2",
        "neuronios_ocultos": "16, 8",
        "csv_token": "",
        "csv_nome": "",
    }


def _converter_int(valor: str, campo: str) -> int:
    try:
        return int(str(valor).strip())
    except (TypeError, ValueError) as exc:
        raise ErroValidacaoModelo(f"Valor inválido para '{campo}': informe um número inteiro.") from exc


def _converter_float(valor: str, campo: str) -> float:
    try:
        texto = str(valor).strip().replace(",", ".")
        num = float(texto)
    except (TypeError, ValueError) as exc:
        raise ErroValidacaoModelo(f"Valor inválido para '{campo}': informe um número válido.") from exc
    return num


@app.route("/", methods=["GET", "POST"])
def index():
    form_state = _estado_inicial_formulario()
    erro: str | None = None
    resultado = None
    graficos = None
    dados = None
    csv_atributos: list[str] = []

    if request.method == "POST":
        for chave in list(form_state.keys()):
            if chave in request.form:
                form_state[chave] = request.form[chave].strip()

        token_atual = form_state.get("csv_token") or ""
        csv_atual = _obter_csv_cache(token_atual) if token_atual else None
        if csv_atual is not None:
            form_state["csv_nome"] = csv_atual["nome"]
            csv_atributos = list(csv_atual["nomes_atributos"])

        try:
            arquivo = request.files.get("arquivo_csv")
            if arquivo and arquivo.filename:
                conteudo = arquivo.read()
                csv_atual = parse_csv_upload(conteudo, arquivo.filename)
                token_atual = _guardar_csv_cache(csv_atual)
                form_state["csv_token"] = token_atual
                form_state["csv_nome"] = csv_atual["nome"]
                csv_atributos = list(csv_atual["nomes_atributos"])
            elif form_state["base"] == "csv" and csv_atual is None:
                csv_atual = _obter_csv_cache(None)
                if csv_atual is not None:
                    form_state["csv_nome"] = csv_atual["nome"]
                    csv_atributos = list(csv_atual["nomes_atributos"])

            taxa = _converter_float(form_state["taxa_aprendizado"], "Taxa de aprendizado")
            epocas = _converter_int(form_state["epocas"], "Épocas")
            lote = _converter_int(form_state["tamanho_lote"], "Tamanho do lote")
            prop_teste_pct = _converter_float(form_state["prop_teste"], "Proporção de teste (%)")
            semente = _converter_int(form_state["semente"], "Semente aleatória")
            attr_x = _converter_int(form_state.get("attr_x", "0"), "Atributo X")
            attr_y = _converter_int(form_state.get("attr_y", "1"), "Atributo Y")
            n_camadas_ocultas = _converter_int(
                form_state.get("n_camadas_ocultas", "2"), "Camadas ocultas do MLP"
            )

            dados = carregar_e_preparar_dados(
                chave_base=form_state["base"],
                prop_teste=prop_teste_pct / 100.0,
                semente=semente,
                csv_cache=csv_atual,
            )

            config = {
                "modelo": form_state["modelo"],
                "ativacao": form_state["ativacao"],
                "taxa_aprendizado": taxa,
                "epocas": epocas,
                "tamanho_lote": lote,
                "semente": semente,
                "attr_x": attr_x,
                "attr_y": attr_y,
                "n_camadas_ocultas": n_camadas_ocultas,
                "neuronios_ocultos": form_state.get("neuronios_ocultos", "16, 8"),
            }
            resultado = executar_treinamento(dados, config)
            graficos = montar_todos_graficos(dados, resultado)

        except (ErroValidacaoDados, ErroValidacaoModelo) as exc:
            erro = str(exc)
        except Exception as exc:
            erro = f"Erro inesperado ao processar o treinamento: {exc}"

    return render_template(
        "index.html",
        form_state=form_state,
        bases=BASES_EMBUTIDAS,
        ativacoes=NOMES_ATIVACOES,
        atributos_json=json.dumps(_ATRIBUTOS_EMBUTIDOS, ensure_ascii=False),
        csv_atributos_json=json.dumps(csv_atributos, ensure_ascii=False),
        erro=erro,
        dados=dados,
        resultado=resultado,
        graficos=graficos,
    )


if __name__ == "__main__":
    app.run(debug=True)
