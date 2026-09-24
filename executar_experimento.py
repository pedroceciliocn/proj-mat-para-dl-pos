"""Executa o experimento duas luas com a rede neural manual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.rede_neural_manual import RedeNeuralManual


def padronizar(
    x_treino: np.ndarray, x_teste: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Padroniza sem usar informação do conjunto de teste."""
    media = x_treino.mean(axis=0)
    desvio = x_treino.std(axis=0)
    return (x_treino - media) / desvio, (x_teste - media) / desvio, media, desvio


def salvar_graficos(
    rede: RedeNeuralManual,
    historico,
    x: np.ndarray,
    y: np.ndarray,
    destino: Path,
) -> None:
    destino.mkdir(parents=True, exist_ok=True)

    fig, eixo = plt.subplots(figsize=(7, 4))
    eixo.plot(historico.epocas, historico.perdas, color="tab:blue")
    eixo.set(xlabel="Época", ylabel="Entropia cruzada", title="Curva de aprendizagem")
    eixo.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destino / "curva_aprendizagem.png", dpi=160)
    plt.close(fig)

    margem = 0.6
    x0_min, x0_max = x[:, 0].min() - margem, x[:, 0].max() + margem
    x1_min, x1_max = x[:, 1].min() - margem, x[:, 1].max() + margem
    grade_x0, grade_x1 = np.meshgrid(
        np.linspace(x0_min, x0_max, 350), np.linspace(x1_min, x1_max, 350)
    )
    grade = np.column_stack((grade_x0.ravel(), grade_x1.ravel()))
    probabilidades = rede.prever_probabilidade(grade).reshape(grade_x0.shape)

    fig, eixo = plt.subplots(figsize=(7, 5))
    contorno = eixo.contourf(
        grade_x0, grade_x1, probabilidades, levels=np.linspace(0, 1, 21),
        cmap="RdBu_r", alpha=0.65
    )
    eixo.contour(grade_x0, grade_x1, probabilidades, levels=[0.5], colors="black")
    eixo.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="white", s=28)
    eixo.set(title="Fronteira de decisão aprendida", xlabel="Atributo 1", ylabel="Atributo 2")
    fig.colorbar(contorno, ax=eixo, label="P(classe = 1)")
    fig.tight_layout()
    fig.savefig(destino / "fronteira_decisao.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=3_000)
    parser.add_argument("--taxa", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--silencioso", action="store_true")
    args = parser.parse_args()

    x, y = make_moons(n_samples=600, noise=0.22, random_state=args.seed)
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.25, random_state=args.seed, stratify=y
    )
    x_treino, x_teste, media, desvio = padronizar(x_treino, x_teste)

    rede = RedeNeuralManual(
        arquitetura=(2, 8, 4, 1), taxa_aprendizado=args.taxa, seed=args.seed
    )
    historico = rede.treinar(
        x_treino,
        y_treino,
        epocas=args.epocas,
        intervalo=max(1, args.epocas // 10),
        verbose=not args.silencioso,
    )

    acuracia_treino = rede.acuracia(x_treino, y_treino)
    acuracia_teste = rede.acuracia(x_teste, y_teste)
    print(f"\nAcurácia de treino: {acuracia_treino:.2%}")
    print(f"Acurácia de teste:  {acuracia_teste:.2%}")

    destino = Path("resultados")
    x_completo_padronizado = (x - media) / desvio
    salvar_graficos(rede, historico, x_completo_padronizado, y, destino)
    metricas = {
        "arquitetura": list(rede.arquitetura),
        "epocas": args.epocas,
        "taxa_aprendizado": args.taxa,
        "seed": args.seed,
        "acuracia_treino": acuracia_treino,
        "acuracia_teste": acuracia_teste,
        "perda_final_treino": historico.perdas[-1],
    }
    (destino / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Gráficos e métricas salvos em: {destino.resolve()}")


if __name__ == "__main__":
    main()

