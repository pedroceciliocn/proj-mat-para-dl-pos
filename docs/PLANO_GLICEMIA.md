# Plano da segunda etapa: previsão de glicemia

## Correção de terminologia

O valor medido por um glicosímetro ou sensor contínuo é a **glicemia** (ou
concentração de glicose, normalmente em mg/dL). **Índice glicêmico** é uma
característica de alimentos ricos em carboidratos. Portanto, o possível modelo
deve ser descrito como previsão da glicemia futura.

## Formulação inicial sugerida

Construir um modelo de regressão que, em um instante `t`, estime a glicemia em
`t + 30 minutos`. Uma linha do conjunto de dados poderia conter:

- leituras de glicose dos últimos 60–120 minutos;
- minutos desde a última refeição e gramas de carboidratos registrados;
- minutos desde a última aplicação e unidades de insulina;
- hora do dia representada por seno e cosseno;
- glicemia medida 30 minutos depois, usada apenas como alvo.

Uma primeira arquitetura manual poderia ser `n_entradas → 16 → 8 → 1`, com
ativação `tanh` nas camadas ocultas e ativação linear na saída. A função de perda
poderia ser erro quadrático médio (MSE), além de MAE e RMSE para avaliação.

## Cuidados metodológicos indispensáveis

1. **Separação temporal:** dados futuros jamais podem participar do treino que
   prevê o passado. Um embaralhamento aleatório causaria vazamento de dados.
2. **Separação por paciente:** se houver várias pessoas, relatar se a avaliação
   é personalizada ou se testa pacientes nunca vistos durante o treino.
3. **Valores ausentes:** intervalos grandes sem leitura não devem ser
   preenchidos sem uma regra explícita e justificável.
4. **Unidades e fusos:** padronizar mg/dL versus mmol/L, horário e frequência de
   amostragem antes de criar janelas temporais.
5. **Baselines:** comparar a rede ao palpite simples de que a glicemia futura
   será igual à última leitura. Uma rede só é útil experimentalmente se superar
   esse baseline fora da amostra.
6. **Privacidade:** dados exportados de sensores e aplicativos são dados
   pessoais sensíveis. Devem ser anonimizados e armazenados com autorização.

## Etapas propostas

1. Concluir e apresentar a classificação de duas luas, que atende diretamente
   ao enunciado e facilita a demonstração da retropropagação.
2. Definir uma fonte de dados devidamente licenciada ou obter uma exportação
   anonimizada e autorizada.
3. Fazer análise exploratória, documentar cada variável e definir exatamente o
   horizonte da previsão.
4. Adaptar a última camada e a perda da rede manual para regressão.
5. Avaliar em um período futuro separado e comparar com o baseline.

## Limite de uso

Essa extensão deve permanecer um estudo acadêmico. Erros em previsões de
glicemia podem causar dano grave se usados para alterar dose de insulina,
alimentação ou tratamento. Qualquer uso clínico exigiria validação, supervisão
profissional e requisitos regulatórios que estão fora do escopo deste projeto.

