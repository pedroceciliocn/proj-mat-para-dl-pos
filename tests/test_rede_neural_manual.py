import numpy as np

from src.rede_neural_manual import RedeNeuralManual


def test_formatos_da_saida_e_dos_parametros():
    rede = RedeNeuralManual(arquitetura=(2, 3, 1), seed=1)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])

    probabilidades = rede.prever_probabilidade(x)

    assert probabilidades.shape == (2,)
    assert np.all((probabilidades >= 0.0) & (probabilidades <= 1.0))
    assert rede.pesos[0].shape == (2, 3)
    assert rede.pesos[1].shape == (3, 1)


def test_gradiente_analitico_confere_com_gradiente_numerico():
    rede = RedeNeuralManual(arquitetura=(2, 3, 1), seed=7)
    x = np.array([[0.2, -0.4], [0.7, 0.1]])
    y = np.array([[1.0], [0.0]])
    _, ativacoes = rede._propagacao_direta(x)
    grad_pesos, _ = rede._retropropagacao(y, ativacoes)

    camada, linha, coluna = 0, 1, 2
    original = rede.pesos[camada][linha, coluna]
    epsilon = 1e-6
    rede.pesos[camada][linha, coluna] = original + epsilon
    perda_mais = rede._perda(y, rede._propagacao_direta(x)[0])
    rede.pesos[camada][linha, coluna] = original - epsilon
    perda_menos = rede._perda(y, rede._propagacao_direta(x)[0])
    rede.pesos[camada][linha, coluna] = original

    gradiente_numerico = (perda_mais - perda_menos) / (2 * epsilon)
    assert np.isclose(
        grad_pesos[camada][linha, coluna], gradiente_numerico, rtol=1e-4, atol=1e-6
    )


def test_treinamento_aprende_problema_xor():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    rede = RedeNeuralManual((2, 6, 1), taxa_aprendizado=0.2, seed=3)

    historico = rede.treinar(x, y, epocas=4_000, intervalo=1_000, verbose=False)

    assert historico.perdas[-1] < historico.perdas[0]
    assert rede.acuracia(x, y) == 1.0

