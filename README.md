# Predição de lesões em corredores competitivos com Aprendizado de Máquina

**Disciplina:** Aprendizado de Máquina para Saúde — Projeto de Aplicação
**Aluno:** Gabriel Abreu Alves

**Notebook de demonstração:** [`AMS_predicao_lesoes_corredores.ipynb`](AMS_predicao_lesoes_corredores.ipynb)

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L1ilk9v1Pwkg51JwN945SMdXtxdP-5Ls?usp=sharing)

---

## 1. Motivação / Objetivo

Lesões por sobrecarga são uma das principais causas de interrupção de treinos e competições em corredores de meia e longa distância. Elas surgem, em grande parte, de desequilíbrios entre a carga de treino aplicada e a capacidade de recuperação do atleta. Treinadores registram essa carga diariamente, mas costumam interpretá-la de forma subjetiva. Um modelo que transforme o log de treino em um **escore de risco de lesão de curto prazo** pode apoiar decisões de ajuste de carga antes que a lesão aconteça.

**Pergunta de pesquisa:** a partir da carga de treino e das percepções subjetivas recentes, é possível estimar o risco de um corredor se lesionar, generalizando para **atletas não vistos no treinamento**? Testamos duas formas de descrever esse histórico, propostas por Lövdal et al. (2021):

- **Abordagem por dia:** o detalhe de cada um dos **7 dias anteriores**.
- **Abordagem por semana:** resumos de cada uma das **3 semanas anteriores**.

**Objetivos específicos:**

1. Formular o problema como **classificação binária** com unidade de análise *dia-atleta*, nas abordagens por dia e por semana.
2. Comparar modelos de complexidade crescente: baseline de classe majoritária, regressão logística, Random Forest, XGBoost, XGBoost *bagged* com subamostragem (estratégia de Lövdal et al., 2021) e uma rede neural MLP.
3. Avaliar com divisão **agrupada por atleta**, para evitar vazamento de informação, e medir o efeito da **normalização por atleta**.
4. Comparar as abordagens **por dia** e **por semana**.

**Público beneficiado:** treinadores, fisiologistas e departamentos médicos de equipes de atletismo.
**Uso pretendido:** ferramenta de triagem para sinalizar dias de risco elevado e motivar a revisão da carga de treino. **Não** é um instrumento diagnóstico.

## 2. Descrição do Dataset

### 2.1 Descrição dos dados

| Item | Valor |
|---|---|
| Fonte | *Replication Data for: Injury Prediction in Competitive Runners With Machine Learning* (Lövdal, Den Hartigh & Azzopardi, 2021), DataverseNL |
| População | 74 corredores de meia e longa distância (800 m à maratona) de uma equipe holandesa de alto nível; 27 mulheres e 47 homens |
| Período | 2012 a 2019 |
| Acesso | Público |
| Variável-alvo | `injury` (1 = lesão; 0 = sem lesão) |
| Identificação | IDs dos atletas mascarados |

O dataset traz dois arquivos, um para cada abordagem. Nos dois, cada linha é um dia de um atleta:

| | Abordagem por dia | Abordagem por semana |
|---|---|---|
| Arquivo | `day_approach_maskedID_timeseries.csv` | `week_approach_maskedID_timeseries.csv` |
| Janela de histórico | 7 dias anteriores | 3 semanas anteriores |
| Variáveis de entrada | 10 variáveis diárias × 7 dias | 22 variáveis semanais × 3 semanas |
| Observações | 42.766 | 42.798 |
| Eventos de lesão | 583 | 575 |
| Prevalência | 1,36% | 1,34% |

**Abordagem por dia.** As 10 variáveis diárias combinam **carga externa** e **percepções subjetivas**:

- **Carga externa:** número de sessões, quilometragem total, quilometragem por zona de intensidade (Z3–4 e Z5/T1/T2), quilometragem de *sprint*, treino de força e horas de treino alternativo.
- **Percepções subjetivas:** esforço percebido, sucesso do treino e recuperação.

A coluna sem sufixo é o dia mais recente, e os sufixos `.1` a `.6` são os dias anteriores.

**Abordagem por semana.** As 22 variáveis resumem cada semana: volumes totais e máximos diários de quilometragem (total e por zona de intensidade), contagens de sessões, dias de descanso e sessões intensas, treino de força e alternativo, e média, mínimo e máximo das percepções de esforço, sucesso do treino e recuperação. Os sufixos `.1` e `.2` indicam as semanas anteriores.

### 2.2 Etapas de preparação dos dados / pré-processamento

1. **Identificação das colunas:** rótulo (`injury`, binarizado), grupo (`Athlete ID`) e data (`Date`). Todas as demais colunas numéricas são entradas.
2. **Remoção de linhas duplicadas.**
3. **Normalização por atleta** (configuração principal): cada variável é convertida em *z-score* com a média e o desvio-padrão do próprio atleta. Assim, cada corredor é comparado com o próprio histórico, e não com a equipe. A transformação não usa o rótulo. Também foi rodada uma configuração **sem normalização**, para comparação.
4. **Imputação de ausentes** pela mediana do treino e **padronização** (regressão logística e MLP), dentro de *pipelines* ajustados só no conjunto de treino.
5. **Divisão por atleta**, estratificada pelo rótulo (`StratifiedGroupKFold`). Nenhum atleta aparece em mais de um conjunto:
   - **Teste:** cerca de 20% dos atletas.
   - **Validação:** cerca de 20% dos atletas restantes, usados para *early stopping* e escolha do limiar.
   - **Treino:** os demais atletas.
6. **Tratamento do desbalanceamento:** pesos de classe (regressão logística, Random Forest, XGBoost e MLP) ou subamostragem da classe negativa (Bagged XGBoost).

### 2.3 Link para a versão disponibilizada do dataset

- **Fonte original:** DataverseNL — [https://doi.org/10.34894/UWU9PV](https://doi.org/10.34894/UWU9PV)
- **Cópias usadas neste projeto:** [`data/day_approach_maskedID_timeseries.csv`](data/day_approach_maskedID_timeseries.csv) e [`data/week_approach_maskedID_timeseries.csv`](data/week_approach_maskedID_timeseries.csv)

## 3. Modelo de aprendizado de máquina

### 3.1 Formalização matemática

**Problema.** Cada observação é um dia de um atleta, descrito pelo histórico de treino anterior: 7 dias detalhados (abordagem por dia) ou 3 semanas resumidas (abordagem por semana). O rótulo indica se houve lesão (1) ou não (0). O modelo recebe essas variáveis e devolve um **escore de risco** entre 0 e 1. O objetivo é que dias com lesão recebam escores mais altos que dias sem lesão.

**Normalização por atleta.** Cada variável é transformada em quantos desvios-padrão ela está acima ou abaixo da média **do próprio atleta**. Assim, "60 km na semana" pode ser carga normal para um corredor e carga alta para outro.

**Modelos comparados:**

| Modelo | Como funciona | Tratamento do desbalanceamento |
|---|---|---|
| Classe majoritária | Prevê sempre "sem lesão"; serve de referência mínima | — |
| Regressão logística | Soma ponderada das variáveis transformada em probabilidade; regularização L2 (C = 0,1) | Pesos de classe |
| Random Forest | Média de 300 árvores de decisão, cada uma treinada com uma amostra diferente dos dados | Pesos de classe |
| XGBoost | Árvores rasas adicionadas uma a uma, cada nova árvore corrigindo os erros das anteriores; o número de árvores é definido por *early stopping* no AUROC de validação | Peso maior para a classe "lesão" |
| Bagged XGBoost (Lövdal et al., 2021) | Média de 25 modelos XGBoost; cada um vê todas as lesões e uma amostra aleatória de dias sem lesão, na proporção 1:1 | Subamostragem |
| MLP | Rede neural com uma camada oculta de 32 neurônios, *dropout* de 0,5 e *early stopping* | Peso maior para a classe "lesão" na função de perda |

**Regra de decisão.** Para transformar o escore em alerta ("risco" / "sem risco"), usamos um limiar escolhido no conjunto de validação. Esse limiar faz com que 80% dos dias sem lesão fiquem abaixo dele (especificidade-alvo de 80%). Todos os modelos usam o mesmo critério, então operam em pontos comparáveis.

### 3.2 Método de validação

- **Validação externa por atleta (*hold-out*):** 48 atletas para treino, 12 para validação e 14 para teste, em cada abordagem. O teste é usado **uma única vez**, ao final.
- **Validação cruzada estratificada e agrupada por atleta (5 folds)** sobre todo o dataset, com os mesmos hiperparâmetros. No XGBoost, o *early stopping* é feito **dentro de cada fold**, com uma validação interna também separada por atleta. Como o teste tem poucos atletas, a **média da CV** é a estimativa principal de desempenho.
- **Intervalo de confiança de 95% da AUROC no teste**, por *bootstrap* que reamostra **atletas** (1.000 repetições). Isso respeita a dependência entre dias do mesmo atleta.
- **As duas abordagens e as duas configurações** (com e sem normalização por atleta) foram avaliadas com o mesmo protocolo e os mesmos hiperparâmetros.
- **Hiperparâmetros fixados a priori**, sem busca extensiva, para não sobreajustar a uma validação com poucos eventos. Os valores estão no notebook (seção 2.4) e em `results/hiperparametros_*.json`.

### 3.3 Medidas de desempenho

| Medida | Definição | Papel |
|---|---|---|
| **AUROC** | Probabilidade de um dia com lesão receber escore maior que um dia sem lesão | Métrica principal (discriminação) |
| **AUPRC** | Área sob a curva precisão–revocação | Informativa sob forte desbalanceamento; o acaso vale a prevalência |
| **Lift da AUPRC** | AUPRC ÷ prevalência | Quantas vezes melhor que o acaso; permite comparar conjuntos com prevalências diferentes |
| Sensibilidade | TP / (TP + FN) | Fração das lesões sinalizadas |
| Especificidade | TN / (TN + FP) | Fração dos dias sem lesão não sinalizados |
| Precisão (VPP) | TP / (TP + FP) | Fração dos alertas que viram lesão |
| F1 | 2 · VPP · Sens / (VPP + Sens) | Equilíbrio entre precisão e sensibilidade |

O *Brier score* não é reportado. Os pesos de classe e a subamostragem deslocam as probabilidades de propósito, então as saídas são escores de **ranqueamento**, não probabilidades calibradas.

## 4. Avaliação

### 4.1 Amostras usadas para treinamento, validação e teste

| Conjunto | Atletas | Por dia: observações | Por dia: lesões (prevalência) | Por semana: observações | Por semana: lesões (prevalência) |
|---|---:|---:|---:|---:|---:|
| Treino | 48 | 28.184 | 372 (1,32%) | 25.960 | 312 (1,20%) |
| Validação | 12 | 6.019 | 81 (1,35%) | 8.050 | 131 (1,63%) |
| Teste | 14 | 8.563 | 130 (1,52%) | 8.788 | 132 (1,50%) |
| **Total** | **74** | **42.766** | **583 (1,36%)** | **42.798** | **575 (1,34%)** |

Na validação cruzada, cada fold de teste tem cerca de 15 atletas, com prevalência entre 0,8% e 1,9%.

### 4.2 Medidas de desempenho

**Validação cruzada agrupada por atleta (5 folds): AUROC, média ± desvio-padrão.** É a estimativa principal.

| Modelo | Por dia, com norm. | Por dia, sem norm. | Por semana, com norm. | Por semana, sem norm. |
|---|---:|---:|---:|---:|
| **Bagged XGBoost** | **0,654 ± 0,027** | **0,637 ± 0,030** | 0,625 ± 0,047 | 0,626 ± 0,021 |
| Regressão logística | 0,626 ± 0,028 | 0,614 ± 0,031 | 0,609 ± 0,032 | 0,617 ± 0,016 |
| Random Forest | 0,624 ± 0,026 | 0,592 ± 0,027 | 0,625 ± 0,068 | **0,629 ± 0,034** |
| XGBoost | 0,611 ± 0,044 | 0,594 ± 0,042 | 0,606 ± 0,041 | 0,618 ± 0,041 |

**Validação cruzada: lift da AUPRC (AUPRC ÷ prevalência), média ± desvio-padrão.**

| Modelo | Por dia, com norm. | Por dia, sem norm. | Por semana, com norm. | Por semana, sem norm. |
|---|---:|---:|---:|---:|
| Bagged XGBoost | 1,91 ± 0,42 | **1,64 ± 0,21** | 2,04 ± 0,34 | **1,56 ± 0,29** |
| Regressão logística | **2,00 ± 0,31** | 1,55 ± 0,14 | **2,22 ± 0,78** | 1,48 ± 0,20 |
| Random Forest | 1,55 ± 0,19 | 1,38 ± 0,16 | 1,74 ± 0,67 | 1,54 ± 0,29 |
| XGBoost | 1,75 ± 0,39 | 1,55 ± 0,48 | 1,86 ± 0,68 | 1,55 ± 0,37 |

**Conjunto de teste, abordagem por dia com normalização por atleta (14 atletas, 130 lesões).** Limiar escolhido na validação para especificidade-alvo de 80%:

| Modelo | AUROC [IC 95%] | AUPRC | Lift | Sens. | Espec. | VPP | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Classe majoritária | 0,500 | 0,015 | 1,00 | 0,000 | 1,000 | — | 0,000 |
| Regressão logística | 0,631 [0,585; 0,685] | 0,033 | 2,17 | 0,354 | 0,802 | 0,027 | 0,050 |
| Random Forest | 0,672 [0,592; 0,738] | 0,029 | 1,88 | 0,277 | 0,887 | 0,036 | 0,064 |
| XGBoost | 0,683 [0,598; 0,753] | 0,035 | 2,34 | 0,285 | 0,897 | 0,041 | 0,072 |
| **Bagged XGBoost** | **0,699 [0,637; 0,760]** | **0,042** | **2,74** | 0,323 | 0,860 | 0,034 | 0,062 |
| MLP | 0,656 [0,598; 0,718] | 0,039 | 2,56 | 0,385 | 0,783 | 0,027 | 0,050 |

**Conjunto de teste: AUROC das demais configurações.**

| Modelo | Por dia, sem norm. | Por semana, com norm. | Por semana, sem norm. |
|---|---:|---:|---:|
| Regressão logística | 0,616 | 0,567 | 0,614 |
| Random Forest | 0,580 | 0,586 | 0,577 |
| XGBoost | 0,593 | 0,604 | 0,567 |
| Bagged XGBoost | **0,666** | **0,606** | 0,597 |
| MLP | 0,621 | 0,601 | **0,623** |

**Observações sobre a avaliação:**

- Na abordagem por dia, o conjunto de teste coincide com o fold 1 da validação cruzada (mesma semente), que foi o fold mais favorável. Por isso a AUROC no teste (0,699) fica acima da média da CV (0,654).
- No XGBoost isolado, o *early stopping* dentro de cada fold escolheu de 1 a 642 árvores (por dia) e de 2 a 251 (por semana). Com cerca de 70 a 130 lesões na validação interna, o ponto de parada é instável, o que explica a variabilidade desse modelo.
- Na abordagem por semana, a variação entre folds é maior: a AUROC de um mesmo modelo vai de 0,55 a 0,71 conforme os atletas do fold.
- As curvas por iteração estão no notebook: perda e AUROC por árvore no XGBoost, desempenho por número de *bags* e de árvores, e perda por época na MLP. Os valores numéricos estão em `results/`.

## 5. Conclusão

- **A abordagem por dia foi melhor que a por semana.** O melhor resultado diário (Bagged XGBoost com normalização, AUROC de 0,654 na CV) supera todas as configurações semanais (de 0,606 a 0,629). Isso concorda com Lövdal et al. (2021), que também encontraram vantagem para a abordagem diária (0,724 contra 0,678). O detalhe dos últimos dias parece carregar mais sinal do que resumos de semanas inteiras.
- **Na abordagem por dia, o Bagged XGBoost foi o modelo mais consistente.** Teve a maior AUROC média nas duas configurações e foi o melhor em 4 dos 5 folds com normalização e em 5 dos 5 sem. Ainda assim, os intervalos de confiança se sobrepõem, e na AUPRC a regressão logística é competitiva (lift de 2,00 contra 1,91).
- **Na abordagem por semana, nenhum modelo se destacou.** As médias ficaram todas entre 0,606 e 0,629, com desvios-padrão maiores que as diferenças entre elas.
- **A agregação por *bagging* estabiliza o modelo.** O XGBoost isolado, dependente de um *early stopping* ruidoso, ficou abaixo do Bagged XGBoost em todas as configurações.
- **A normalização por atleta ajuda mais na abordagem por dia.** Por dia, ela melhorou a AUROC de todos os modelos (cerca de 0,01 a 0,03). Por semana, a AUROC praticamente não mudou, mas o lift da AUPRC subiu em todos os modelos (Bagged XGBoost de 1,56 para 2,04). Ou seja, a normalização ajuda a concentrar as lesões entre os dias de maior risco.
- **Comparação com a literatura:** os resultados ficam abaixo dos reportados por Lövdal et al. (2021), mas mantêm a mesma ordem entre as abordagens. A diferença provavelmente vem do protocolo de avaliação: aqui a validação é estritamente separada por atleta.
- **Utilidade prática:** com especificidade em torno de 86%, o Bagged XGBoost diário detecta cerca de 32% das lesões, com precisão de 3,4%. Isso equivale a aproximadamente 1 lesão real a cada 29 alertas. O modelo serve como **sinal de risco** para revisar a carga, não como preditor confiável de lesões individuais.

**Limitações:**

- Poucos eventos de lesão (cerca de 580) e uma única equipe, o que limita a generalização.
- A normalização por atleta usa estatísticas de todo o período do atleta, então parte do ganho pode ser otimista.
- As saídas não são probabilidades calibradas.
- O limiar de 80% de especificidade não reflete necessariamente o custo clínico de falsos negativos e falsos positivos.
- Fatores relevantes ficam de fora, como histórico de lesões, sono e biomecânica.

**Trabalhos futuros:**

- Normalização causal (só com dados passados do atleta).
- Avaliação temporal (treinar no passado e testar no futuro).
- Calibração das probabilidades.
- Combinação das duas abordagens (detalhe dos últimos dias + resumo das semanas anteriores).
- Validação externa em outros esportes e equipes, como o dataset SoccerMon (futebol).

## Referências

- Lövdal, S. S., Den Hartigh, R. J. R., & Azzopardi, G. (2021). Injury Prediction in Competitive Runners With Machine Learning. *International Journal of Sports Physiology and Performance*, 16(10), 1522–1531. https://doi.org/10.1123/ijspp.2020-0518
- Lövdal, S. S., Den Hartigh, R. J. R., & Azzopardi, G. (2021). *Replication Data for: Injury Prediction In Competitive Runners With Machine Learning*. DataverseNL. https://doi.org/10.34894/UWU9PV
- Rossi, A., Pappalardo, L., Cintia, P., Iaia, F. M., Fernàndez, J., & Medina, D. (2018). Effective injury forecasting in soccer with GPS training data and machine learning. *PLoS ONE*, 13(7), e0201264. https://doi.org/10.1371/journal.pone.0201264

## Estrutura do repositório

```
├── README.md
├── AMS_predicao_lesoes_corredores.ipynb   # notebook de demonstração (Parte 2)
├── data/
│   ├── day_approach_maskedID_timeseries.csv
│   └── week_approach_maskedID_timeseries.csv
└── results/                               # tabelas geradas pelo notebook
    ├── splits_{day,week}_*.csv
    ├── metricas_teste_{day,week}_*.csv
    ├── cv_folds_{day,week}_*.csv / cv_resumo_{day,week}_*.csv
    ├── xgb_curvas_validacao_{day,week}_*.csv
    ├── mlp_historico_{day,week}_*.csv
    └── hiperparametros_{day,week}_*.json
```