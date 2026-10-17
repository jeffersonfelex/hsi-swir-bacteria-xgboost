# Validação do repositório × material (Drive)

Fontes: *Apresentacao_Oral_FINAL.pptx* e *Guia de estudo — HSI-SWIR + XGBoost* (05/10/2026).
Conferido contra o código em `chore/baseline-pipeline-v2` e o dataset local em `features_selected/`.

## Confere

| Item do material | Código |
|---|---|
| ROI → 4 branches paralelos (raw, MSC, Log, deriv) | `main.py` |
| SG 1ª derivada, janela 15, grau 2 | `etl/derivative_transform.py` |
| Rótulo pela procedência (ATCC13/16/27 = 1) | `etl/label_by_atcc.py` |
| 23 atributos (desc_* e *_area_norm removidos) | `ml/train.py`; `n_features_originais=23` |
| StandardScaler → PCA 95% → XGBoost | `ml/train.py` |
| Grid de 324 combinações (3·3·3·2·2·3) | `ml/train.py` |
| Split por imagem, estratificado por classe | `ml/split_dataset.py` |
| CV por grupo (StratifiedGroupKFold) | `ml/train.py` |
| Recall primário, AUPRC secundária; métricas por pixel e imagem | `ml/train.py`, `ml/evaluate.py` |

## Divergências (o material está desatualizado em relação ao código/dados)

| Material diz | Repositório mostra |
|---|---|
| 223.762 px; 42.198 resistentes / 181.564 sensíveis; prevalência ≈ 0,19 | **1.059.017 px; 134.378 / 924.639; prevalência 0,127** |
| 5 folds, só 2 informativos | CV **adaptativa**: `n_splits = min(5, nº de imagens resistentes no treino)` = **2** (13 imagens de treino) |
| MSC e Log(1/R) melhores (recall ≈ 0,34–0,50; AUPRC ≈ 0,70) | Última execução (`ml/results/comparacao_pipelines.csv`): **Log 0,181 > Deriv 0,153 > MSC 0,137 > Raw 0,04**; AUPRC 0,12–0,19, ≈ prevalência (sem sinal acima do acaso) |
| MI e XGB-FI "provavelmente escolheram o mesmo subconjunto" | Causa objetiva: `MI_TOP_N=100` e `XGB_TOP_N=50` ≥ nº de atributos, então **nenhum filtra** em msc/log/deriv. Agora `main.py` emite aviso |
| 3 de 5 folds sem resistentes, recall registrado como 0 | `train.py` agora grava **NaN** (indefinido) e exclui das médias |
| IC95% t com 5 folds | Com n=2 o IC saía [−1,83; 2,19]. Agora NaN se n<3 e limitado a [0,1] |

**Ação sugerida:** antes de apresentar/submeter, refazer slides 6, 8 e 9 e a conclusão com os números atuais
(a conclusão qualitativa — "não é possível afirmar" — segue válida e até se reforça).

## Pendências do guia — situação

| Pendência do guia | Situação |
|---|---|
| Confirmar que a seleção MI/XGB-FI usa só treino | **Corrigido.** Antes, `main.py` selecionava sobre *todas* as imagens (incl. teste) antes do split — vazamento de rótulo em **raw** (256→100→50 bandas). Agora `main.py` só grava `dataset_<branch>.csv` (4 arquivos, sem seleção); `ml/split_dataset.py` divide por imagem uma vez por branch; a seleção é a etapa `SelecaoMIXGB` (`ml/selection.py`) do `Pipeline` (`selecao → scaler → PCA → XGBoost`), **reajustada a cada fold** só com o fold de treino e, no refit final, só com o treino externo. O `pipeline_*.joblib` carrega as colunas escolhidas e `ml/evaluate.py` apenas as aplica ao teste. Coberto por testes sintéticos (`tests/test_validation.py`). **Efeito nos números:** os resultados versionados em `ml/results` foram gerados com a seleção antiga — refazer `split → train → evaluate`; raw tende a piorar (menos otimista) |
| MSC com média só do treino | MSC usa a média **da própria imagem** (por ROI), não global. Não há vazamento de rótulo, mas difere do descrito |
| `scale_pos_weight` usado? | **Não** é usado em `train.py` (razão atual ≈ 6,9 : 1, não 4,3) |
| Grid Search refit por recall | `refit="recall"` favorece limiar permissivo; considerar AUPRC e CV aninhada |
| CV selecionando e reportando no mesmo split | Estimativa otimista; CV aninhada é o próximo passo |
| `checar_folds`, `avaliar_teste_externo`, `analises_complementares.py`, importância por permutação | **Não existem** no repo (equivalentes parciais: `fold_scores_*.csv`, `ml/evaluate.py`) |
| Teste externo preenchido (slide 9) | Rodar `python run.py evaluate`; `ml/reports/` não é versionado |
| Espécie, CAAE, n de isolados, antibiograma, agência | Não derivável do código — dependem do orientador |

## Outras correções nesta série de branches

- `evaluate.py` reajustava scaler/PCA por conta própria; podia divergir do treino (principalmente com
  `IC2025_MAX_PIXELS_PER_IMG`). Agora carrega o `pipeline_*.joblib` salvo pelo `train.py`.
- `.pyc` e `.claude/settings.local.json` estavam versionados; `ml/data` (~5 GB) segue ignorado.
- `requirements.txt` listava keras/catboost/lightgbm/sqlalchemy etc. que não são importados.
- `scripts/paper_figures.py` tinha prevalência hardcoded (0,13) diferente do dataset.
- 23 testes em `tests/`.
- Seleção de atributos movida para depois do split (ver acima). Detalhes de implementação: MI/XGB-FI são
  ajustados em até `IC2025_SELECAO_MAX_AMOSTRAS` pixels (padrão 50.000, subamostra determinística só do treino; `0` = todos)
  para não repetir MI (kNN) em ~1 M pixels a cada fold; o `Pipeline(memory=...)` evita refazê-la a cada combinação do grid.
  O filtro `desc_*`/`*_area_norm` agora precede a seleção. Em msc/log/deriv (23 atributos < `MI_TOP_N`) a seleção continua sem filtrar nada.
  Artefatos novos: `ml/data/train_<branch>.csv`/`test_<branch>.csv` (um por branch) e `ml/results/features_selected_<pipeline>.csv`.

**Nota:** os modelos/resultados versionados em `ml/` vêm da execução anterior; rode
`python run.py train` para gerar `pipeline_*.joblib` e CSVs com a nova lógica de NaN.
