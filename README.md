# Rede neural manual para classificação de dados não lineares

Projeto da disciplina **Matemática para Ciência de Dados**, desenvolvido a
partir do `exemplo4.py` da aula de 21/08/2026. O objetivo é mostrar, de forma
didática, a matemática da propagação direta e da retropropagação sem utilizar
frameworks prontos de redes neurais.

## O que foi modificado em relação ao exemplo

| Elemento | `exemplo4.py` | Este projeto |
|---|---:|---:|
| Arquitetura manual | 2 → 2 → 1 | **2 → 8 → 4 → 1** |
| Camadas ocultas | 1 | **2** |
| Ativação oculta | sigmoide | **tanh** |
| Ativação de saída | sigmoide | sigmoide |
| Função de perda | erro quadrático | **entropia cruzada binária** |
| Quantidade de exemplos | 100 | **600** |
| Separação treino/teste | não | **sim (75%/25%)** |
| Inicialização | uniforme em [0, 1) | **Xavier** |
| Implementação | operações escalares | **operações matriciais explícitas** |

O conjunto `make_moons` foi mantido nesta primeira etapa para que o efeito das
alterações na rede possa ser comparado ao material da aula. O scikit-learn é
usado apenas para gerar e dividir os dados; ele **não** treina a rede.

## Matemática usada

Para uma camada qualquer, a combinação linear e a ativação são:

```text
Z[l] = A[l-1] W[l] + b[l]
A[l] = tanh(Z[l])                 (camadas ocultas)
A[L] = sigmoid(Z[L])             (camada de saída)
```

A perda para uma amostra de classe `y` e probabilidade prevista `ŷ` é:

```text
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

Com entropia cruzada e sigmoide na saída, o primeiro gradiente da
retropropagação se reduz a `δ[L] = ŷ - y`. Nas camadas ocultas aplica-se a regra
da cadeia e a derivada `tanh'(z) = 1 - tanh²(z)`. Os parâmetros são atualizados
por gradiente descendente:

```text
W := W - taxa * dL/dW
b := b - taxa * dL/db
```

O código correspondente está comentado em
[`src/rede_neural_manual.py`](src/rede_neural_manual.py).

## Requisitos

- Python 3.10 ou mais recente;
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) para gerenciar
  o ambiente e as dependências;
- VS Code com as extensões Python e Jupyter, caso o notebook seja executado
  pelo editor.

## Instalação com uv

Na raiz do projeto, sincronize o ambiente usando as versões registradas no
`uv.lock`:

```bash
uv sync --locked
```

O `uv` cria a `.venv` automaticamente. Não é necessário ativá-la: todo comando
pode ser executado com `uv run`.

## Executando o experimento

Para executar o treinamento completo:

```bash
uv run python executar_experimento.py
```

Os argumentos disponíveis podem ser consultados com
`uv run python executar_experimento.py -h`. Para um teste curto:

```bash
uv run python executar_experimento.py --epocas 500
```

Ao final, a pasta `resultados/` conterá:

- `curva_aprendizagem.png`: evolução da perda no treino;
- `fronteira_decisao.png`: regiões que a rede atribui a cada classe;
- `metricas.json`: configuração e acurácias de treino e teste.

## Executando o notebook no VS Code

Abra `notebooks/projeto_rede_neural_manual.ipynb`, clique em **Select Kernel**,
escolha **Python Environments** e selecione `.venv`. O workspace já contém a
configuração necessária para o VS Code localizar esse interpretador.

Se o ambiente não aparecer, execute **Developer: Reload Window** pela paleta de
comandos ou informe manualmente o caminho `.venv/bin/python`.

Também é possível registrar um kernelspec com um nome próprio:

```bash
uv run python -m ipykernel install --user \
  --name rede-neural-manual \
  --display-name "Python (rede-neural-manual)"
```

Nesse caso, selecione **Python (rede-neural-manual)**. Tanto `.venv` quanto esse
kernelspec apontam para o mesmo ambiente gerenciado pelo `uv`.

## Instalação alternativa com pip

Para ambientes sem `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Google Colab

O notebook autocontido
[`notebooks/projeto_rede_neural_manual.ipynb`](notebooks/projeto_rede_neural_manual.ipynb)
pode ser enviado diretamente ao Colab em **Arquivo → Fazer upload de notebook**.
Depois que o repositório estiver público, também será possível adicionar ao topo
deste README o botão “Open in Colab” apontando para a URL definitiva.

## Testes

```bash
uv run pytest -q
```

Os testes verificam formatos e probabilidades, conferem a retropropagação por
gradiente numérico e confirmam que a rede consegue aprender o problema XOR.
Eles também são executados automaticamente pelo GitHub Actions a cada `push`
ou `pull request`.

## Organização

```text
.
├── .github/workflows/testes.yml        # integração contínua com GitHub Actions
├── .vscode/settings.json               # interpretador padrão do workspace
├── exemplo4.py                         # código original do professor
├── executar_experimento.py             # experimento reproduzível
├── pyproject.toml                       # projeto e dependências para o uv
├── requirements.txt                    # instalação alternativa com pip
├── uv.lock                              # versões reproduzíveis
├── src/rede_neural_manual.py           # rede feita somente com NumPy
├── tests/test_rede_neural_manual.py    # testes automatizados
├── resultados/                         # métricas e gráficos reproduzidos
├── notebooks/
│   └── projeto_rede_neural_manual.ipynb
└── docs/PLANO_GLICEMIA.md              # proposta para a segunda etapa
```

## Próxima etapa: previsão de glicemia

A ideia de prever glicemia a partir do histórico do paciente, carboidratos e
insulina é interessante, mas muda o problema de classificação para previsão
temporal/regressão. O desenho responsável dessa etapa está em
[`docs/PLANO_GLICEMIA.md`](docs/PLANO_GLICEMIA.md). Ela deve ser tratada como
experimento acadêmico, não como ferramenta para orientar doses de insulina.
