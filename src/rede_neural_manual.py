"""Rede neural multicamada implementada somente com NumPy.

O módulo não usa Keras, TensorFlow, PyTorch ou o MLP do scikit-learn. Todas as
etapas da propagação direta, retropropagação e atualização dos parâmetros estão
explícitas para fins didáticos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Historico:
    """Métricas registradas durante o treinamento."""

    epocas: list[int]
    perdas: list[float]
    acuracias: list[float]


class RedeNeuralManual:
    """Perceptron multicamada para classificação binária.

    As camadas ocultas usam tangente hiperbólica e a saída usa sigmoide. A
    função de custo é a entropia cruzada binária. A combinação sigmoide +
    entropia cruzada simplifica o gradiente da última camada para ``y_pred-y``.
    """

    def __init__(
        self,
        arquitetura: tuple[int, ...] = (2, 8, 4, 1),
        taxa_aprendizado: float = 0.05,
        seed: int = 42,
    ) -> None:
        if len(arquitetura) < 2 or arquitetura[-1] != 1:
            raise ValueError("A arquitetura deve ter ao menos entrada e uma saída.")
        if any(n <= 0 for n in arquitetura):
            raise ValueError("Todas as camadas devem possuir ao menos um neurônio.")
        if taxa_aprendizado <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva.")

        self.arquitetura = arquitetura
        self.taxa_aprendizado = taxa_aprendizado
        gerador = np.random.default_rng(seed)

        # Inicialização de Xavier: adequada à ativação tanh.
        self.pesos = []
        self.vieses = []
        for n_entrada, n_saida in zip(arquitetura[:-1], arquitetura[1:]):
            limite = np.sqrt(6.0 / (n_entrada + n_saida))
            self.pesos.append(
                gerador.uniform(-limite, limite, size=(n_entrada, n_saida))
            )
            self.vieses.append(np.zeros((1, n_saida)))

    @staticmethod
    def _sigmoide(z: np.ndarray) -> np.ndarray:
        """Sigmoide numericamente estável."""
        saida = np.empty_like(z, dtype=float)
        positivos = z >= 0
        saida[positivos] = 1.0 / (1.0 + np.exp(-z[positivos]))
        exp_z = np.exp(z[~positivos])
        saida[~positivos] = exp_z / (1.0 + exp_z)
        return saida

    def _propagacao_direta(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Calcula as ativações e guarda valores usados na retropropagação."""
        ativacoes = [x]
        ativacao = x

        for indice, (peso, vies) in enumerate(zip(self.pesos, self.vieses)):
            z = ativacao @ peso + vies
            eh_saida = indice == len(self.pesos) - 1
            ativacao = self._sigmoide(z) if eh_saida else np.tanh(z)
            ativacoes.append(ativacao)

        return ativacao, ativacoes

    @staticmethod
    def _perda(y: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-12
        probabilidades = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return float(
            -np.mean(y * np.log(probabilidades) + (1.0 - y) * np.log(1.0 - probabilidades))
        )

    def _retropropagacao(
        self, y: np.ndarray, ativacoes: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Aplica a regra da cadeia e devolve gradientes de pesos e vieses."""
        tamanho_lote = y.shape[0]
        grad_pesos = [np.empty_like(peso) for peso in self.pesos]
        grad_vieses = [np.empty_like(vies) for vies in self.vieses]

        # Derivada de entropia cruzada binária após uma saída sigmoide.
        delta = ativacoes[-1] - y

        for camada in range(len(self.pesos) - 1, -1, -1):
            grad_pesos[camada] = ativacoes[camada].T @ delta / tamanho_lote
            grad_vieses[camada] = np.mean(delta, axis=0, keepdims=True)

            if camada > 0:
                delta = delta @ self.pesos[camada].T
                # Se a = tanh(z), então da/dz = 1 - a².
                delta *= 1.0 - ativacoes[camada] ** 2

        return grad_pesos, grad_vieses

    def treinar(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epocas: int = 3_000,
        intervalo: int = 100,
        verbose: bool = True,
    ) -> Historico:
        """Treina a rede por gradiente descendente em lote completo."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if x.ndim != 2 or x.shape[1] != self.arquitetura[0]:
            raise ValueError(f"x deve ter formato (n, {self.arquitetura[0]}).")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x e y devem possuir o mesmo número de amostras.")
        if epocas <= 0 or intervalo <= 0:
            raise ValueError("epocas e intervalo devem ser positivos.")

        historico = Historico([], [], [])
        for epoca in range(epocas + 1):
            y_pred, ativacoes = self._propagacao_direta(x)

            if epoca % intervalo == 0 or epoca == epocas:
                perda = self._perda(y, y_pred)
                acuracia = float(np.mean((y_pred >= 0.5) == y))
                historico.epocas.append(epoca)
                historico.perdas.append(perda)
                historico.acuracias.append(acuracia)
                if verbose:
                    print(
                        f"Época {epoca:5d} | perda: {perda:.6f} "
                        f"| acurácia: {acuracia:.2%}"
                    )

            if epoca == epocas:
                break

            grad_pesos, grad_vieses = self._retropropagacao(y, ativacoes)
            for camada in range(len(self.pesos)):
                self.pesos[camada] -= self.taxa_aprendizado * grad_pesos[camada]
                self.vieses[camada] -= self.taxa_aprendizado * grad_vieses[camada]

        return historico

    def prever_probabilidade(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        probabilidades, _ = self._propagacao_direta(x)
        return probabilidades.ravel()

    def prever(self, x: np.ndarray, limiar: float = 0.5) -> np.ndarray:
        if not 0.0 < limiar < 1.0:
            raise ValueError("O limiar deve estar entre 0 e 1.")
        return (self.prever_probabilidade(x) >= limiar).astype(int)

    def acuracia(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.prever(x) == y))

