## Group-Aware Machine Learning for SWIR Hyperspectral Characterization of Bacterial Isolates

Classificação espectral de bactérias com imagens hiperespectrais SWIR (1000–2500 nm, 256 bandas)
e XGBoost, com validação **por imagem** (sem pseudo-replicação de pixels).

> **Status: resultado exploratório.** O pipeline é tecnicamente viável, mas os dados atuais
> (3 imagens resistentes, todas ATCC; 16 imagens sensíveis, todas clínicas) confundem
> *resistência* com *procedência*. Não é possível afirmar que o modelo detecta resistência.
> Veja [docs/VALIDACAO.md](docs/VALIDACAO.md).

### Pipeline

```
ROI (segmentação da colônia)
 ├─ raw   : bandas brutas
 ├─ msc   : Multiplicative Scatter Correction
 ├─ log   : Log(1/R)
 └─ deriv : 1ª derivada Savitzky-Golay (janela 15, grau 2)
      └─ atributos por pixel (spec_ + idx_ = 23 após remover desc_* e *_area_norm)
      └─ 4 datasets completos em features_selected/ (sem seleção)
split por imagem 70/30 (estratificado por classe)
 → seleção MI / XGB-FI (só no treino, refeita a cada fold) → StandardScaler → PCA (95% var.) → XGBoost, Grid Search (324 comb.), StratifiedGroupKFold
 → avaliação única no teste externo (pixel e imagem)
```

### Uso

```bash
pip install -r requirements.txt      # requirements-dev.txt: notebooks e testes
python run.py check                  # verifica ambiente / GPU
python run.py all                    # etl -> split -> train -> evaluate
python -m pytest tests               # testes das garantias de validação
```

Etapas isoladas: `python run.py etl | split | train | evaluate`. Dados HSI em
`dataSet/hsi_original/` (não versionado). Variáveis: `IC2025_BASE_DIR`, `IC2025_GRID=fast`,
`IC2025_MAX_PIXELS_PER_IMG`, `IC2025_SELECAO_MAX_AMOSTRAS`, `IC2025_FORCE_CPU=1`.

### Estrutura

| Caminho | Conteúdo |
|---|---|
| `main.py`, `etl/` | ETL: ROI, transformações, atributos, rotulagem (ATCC = 1) |
| `ml/split_dataset.py` | Split treino/teste por imagem (por branch) |
| `ml/selection.py` | Seleção MI / XGB-FI como etapa do Pipeline (ajustada só no treino) |
| `ml/train.py` | Grid Search + CV por grupo; salva modelo e `pipeline_*.joblib` |
| `ml/evaluate.py` | Teste externo (pixel e imagem) |
| `ml/results/` | CSVs e boxplots da CV |
| `scripts/paper_figures.py` | Figuras do artigo |
| `dags/` | DAG Airflow (opcional) |
| `notebooks/` | Exploração de pré-processamento |
| `docs/VALIDACAO.md` | Conferência código × slides × guia de estudo |
