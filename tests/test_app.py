"""Testes automatizados para validar todos os requisitos RF1 a RF12 da aplicação Flask."""

from __future__ import annotations

import io
import os
import sys
import unittest

import numpy as np

PASTA_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PASTA_APP not in sys.path:
    sys.path.insert(0, PASTA_APP)

from app import app  # noqa: E402
from dados import carregar_e_preparar_dados, parse_csv_upload  # noqa: E402
from modelos import executar_treinamento  # noqa: E402


def _gerar_csv_teste(sep: str = ",", classes_texto: bool = False, n_linhas: int = 40) -> bytes:
    rng = np.random.default_rng(7)
    linhas = [f"f1{sep}f2{sep}f3{sep}classe"]
    for i in range(n_linhas):
        c = i % 2
        f1 = rng.normal(loc=2.0 if c else -2.0, scale=0.7)
        f2 = rng.normal(loc=1.0 if c else -1.0, scale=0.8)
        f3 = rng.normal(loc=0.0, scale=1.0)
        if sep == ";":
            s1, s2, s3 = f"{f1:.3f}".replace(".", ","), f"{f2:.3f}".replace(".", ","), f"{f3:.3f}".replace(".", ",")
        else:
            s1, s2, s3 = f"{f1:.3f}", f"{f2:.3f}", f"{f3:.3f}"
        rotulo = ("Positivo" if c else "Negativo") if classes_texto else str(c)
        linhas.append(f"{s1}{sep}{s2}{sep}{s3}{sep}{rotulo}")
    return "\n".join(linhas).encode("utf-8")


class TestesAplicacaoRedesNeurais(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_get_inicial(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("Configuração do Experimento", html)
        self.assertIn("Treinar modelo e analisar parâmetros", html)

    def test_rf5_sem_vazamento_dados(self) -> None:
        dados_tab = carregar_e_preparar_dados("iris", prop_teste=0.25, semente=42)
        self.assertEqual(dados_tab.tipo, "tabular")
        # Média e desvio do conjunto de treino padronizado devem ser ~0 e ~1
        self.assertTrue(np.allclose(dados_tab.X_treino.mean(axis=0), 0.0, atol=1e-7))
        self.assertTrue(np.allclose(dados_tab.X_treino.std(axis=0), 1.0, atol=1e-7))

        dados_img = carregar_e_preparar_dados("digits", prop_teste=0.25, semente=42)
        self.assertEqual(dados_img.tipo, "imagem")
        self.assertGreaterEqual(float(dados_img.X_treino_img.min()), 0.0)
        self.assertLessEqual(float(dados_img.X_treino_img.max()), 1.0)
        self.assertGreaterEqual(float(dados_img.X_teste_img.min()), 0.0)
        self.assertLessEqual(float(dados_img.X_teste_img.max()), 1.0)

    def test_reprodutibilidade_semente(self) -> None:
        d1 = carregar_e_preparar_dados("iris", 0.25, 42)
        d2 = carregar_e_preparar_dados("iris", 0.25, 42)
        cfg = {
            "modelo": "perceptron",
            "ativacao": "sigmoide",
            "taxa_aprendizado": 0.1,
            "epocas": 25,
            "tamanho_lote": 16,
            "semente": 42,
            "attr_x": 0,
            "attr_y": 2,
        }
        r1 = executar_treinamento(d1, cfg)
        r2 = executar_treinamento(d2, cfg)
        self.assertEqual(r1.acuracia_teste, r2.acuracia_teste)
        self.assertAlmostEqual(r1.perda_final, r2.perda_final, places=10)
        self.assertAlmostEqual(r1.tabela_inicial_final[0]["final"], r2.tabela_inicial_final[0]["final"], places=10)

    def test_perceptron_binario_e_multiclasse(self) -> None:
        for base, ativ in [("iris", "degrau"), ("iris", "sigmoide"), ("iris", "tanh"), ("breast_cancer", "sigmoide")]:
            resp = self.client.post(
                "/",
                data={
                    "modelo": "perceptron",
                    "base": base,
                    "ativacao": ativ,
                    "taxa_aprendizado": "0.1",
                    "epocas": "30",
                    "tamanho_lote": "16",
                    "prop_teste": "25",
                    "semente": "42",
                    "attr_x": "0",
                    "attr_y": "1",
                },
            )
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode("utf-8")
            self.assertNotIn('role="alert"', html)
            self.assertIn("Fronteira de Decisão do Perceptron", html)
            self.assertIn("Δ total", html)
            self.assertIn("Norma da variação dos pesos", html)

    def test_mlp_configuravel(self) -> None:
        for ativ, camadas, neuronios in [
            ("relu", "2", "16, 8"),
            ("leaky_relu", "2", "12"),
            ("tanh", "1", "10"),
            ("sigmoide", "2", "8, 4"),
        ]:
            resp = self.client.post(
                "/",
                data={
                    "modelo": "mlp",
                    "base": "iris",
                    "ativacao": ativ,
                    "taxa_aprendizado": "0.1",
                    "epocas": "25",
                    "tamanho_lote": "16",
                    "prop_teste": "25",
                    "semente": "42",
                    "n_camadas_ocultas": camadas,
                    "neuronios_ocultos": neuronios,
                },
            )
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode("utf-8")
            self.assertNotIn('role="alert"', html)
            self.assertIn("Mapa de Pesos — Inicial vs Final", html)

    def test_cnn_com_visualizacao_filtros(self) -> None:
        resp = self.client.post(
            "/",
            data={
                "modelo": "cnn",
                "base": "digits",
                "ativacao": "relu",
                "taxa_aprendizado": "0.1",
                "epocas": "10",
                "tamanho_lote": "32",
                "prop_teste": "25",
                "semente": "42",
                "num_filtros": "6",
                "tamanho_kernel": "3",
            },
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertNotIn('role="alert"', html)
        self.assertIn("Visualização dos Filtros da CNN", html)
        self.assertIn("Filtro 1", html)

    def test_upload_csv_exclusivo_e_reuso_sem_reiniciar(self) -> None:
        csv_bytes = _gerar_csv_teste(sep=";", classes_texto=True)
        resp1 = self.client.post(
            "/",
            data={
                "modelo": "perceptron",
                "base": "csv",
                "ativacao": "sigmoide",
                "taxa_aprendizado": "0.1",
                "epocas": "20",
                "tamanho_lote": "8",
                "prop_teste": "25",
                "semente": "42",
                "attr_x": "0",
                "attr_y": "1",
                "arquivo_csv": (io.BytesIO(csv_bytes), "minha_base_exclusiva.csv"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp1.status_code, 200)
        html1 = resp1.data.decode("utf-8")
        self.assertNotIn('role="alert"', html1)
        self.assertIn("minha_base_exclusiva.csv", html1)

        # Troca para MLP usando o CSV já em cache sem reenviar arquivo nem reiniciar servidor (RF2)
        resp2 = self.client.post(
            "/",
            data={
                "modelo": "mlp",
                "base": "csv",
                "ativacao": "relu",
                "taxa_aprendizado": "0.1",
                "epocas": "20",
                "tamanho_lote": "8",
                "prop_teste": "25",
                "semente": "42",
                "n_camadas_ocultas": "2",
                "neuronios_ocultos": "10, 6",
            },
        )
        self.assertEqual(resp2.status_code, 200)
        html2 = resp2.data.decode("utf-8")
        self.assertNotIn('role="alert"', html2)
        self.assertIn("minha_base_exclusiva.csv", html2)

    def test_rf12_tratamento_erros_sem_derrubar(self) -> None:
        casos_invalidos = [
            # CNN com base tabular
            {"modelo": "cnn", "base": "iris", "ativacao": "relu", "epocas": "10"},
            # Perceptron com ReLU
            {"modelo": "perceptron", "base": "iris", "ativacao": "relu", "epocas": "20"},
            # MLP com degrau
            {"modelo": "mlp", "base": "iris", "ativacao": "degrau", "epocas": "20"},
            # CNN com degrau
            {"modelo": "cnn", "base": "digits", "ativacao": "degrau", "epocas": "10"},
            # Atributos iguais na fronteira do Perceptron
            {"modelo": "perceptron", "base": "iris", "ativacao": "sigmoide", "epocas": "20", "attr_x": "1", "attr_y": "1"},
            # Épocas acima do limite da CNN
            {"modelo": "cnn", "base": "digits", "ativacao": "relu", "epocas": "35"},
            # Kernel grande demais para 8x8
            {"modelo": "cnn", "base": "digits", "ativacao": "relu", "epocas": "5", "tamanho_kernel": "8"},
            # Camadas e neurônios inconsistentes no MLP
            {"modelo": "mlp", "base": "iris", "ativacao": "relu", "epocas": "10", "n_camadas_ocultas": "3", "neuronios_ocultos": "16, 8"},
        ]
        base_padrao = {
            "taxa_aprendizado": "0.1",
            "tamanho_lote": "16",
            "prop_teste": "25",
            "semente": "42",
            "attr_x": "0",
            "attr_y": "1",
            "n_camadas_ocultas": "2",
            "neuronios_ocultos": "16, 8",
            "num_filtros": "8",
            "tamanho_kernel": "3",
        }
        for caso in casos_invalidos:
            payload = {**base_padrao, **caso}
            resp = self.client.post("/", data=payload)
            self.assertEqual(resp.status_code, 200)
            self.assertIn('role="alert"', resp.data.decode("utf-8"), f"Esperava erro para {caso}")

    def test_rf12_csv_mal_formatado(self) -> None:
        csvs_ruins = [
            (b"", "vazio.csv"),
            (b"a,b,c\n1,2,0\n3,4,1\n", "poucas_linhas.csv"),
            (b"a,b,c\n1,x,0\n" * 12, "nao_numerico.csv"),
            (b"a,b,c\n1,2,0\n1,2\n" * 6, "colunas_inconsistentes.csv"),
            (b"a,b,c\n1,2,0\n" * 12, "uma_classe.csv"),
            (_gerar_csv_teste(), "arquivo.txt"),
        ]
        for conteudo, nome in csvs_ruins:
            with self.assertRaises(ValueError, msg=f"Deveria falhar para {nome}"):
                parse_csv_upload(conteudo, nome)

        # Também testa pela rota HTTP sem derrubar o servidor
        resp = self.client.post(
            "/",
            data={
                "modelo": "perceptron",
                "base": "csv",
                "ativacao": "sigmoide",
                "taxa_aprendizado": "0.1",
                "epocas": "10",
                "tamanho_lote": "8",
                "prop_teste": "25",
                "semente": "42",
                "arquivo_csv": (io.BytesIO(b"a,b,c\n1,texto,0\n" * 12), "quebrado.csv"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn('role="alert"', resp.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
