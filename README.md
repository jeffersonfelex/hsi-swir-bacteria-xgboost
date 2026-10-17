# Group-Aware Machine Learning for SWIR Hyperspectral Characterization of Bacterial Isolates

**Classificação espectral de bactérias multirresistentes com Gradient Boosting aplicado a imagens hiperespectrais SWIR**

Iniciação científica (IC 2025) · Laboratório de Computação Científica (LCC) · PUC Goiás

| | |
|---|---|
| **Autor** | Jefferson Felex de Faria |
| **Orientador** | Prof. Clarimar José Coelho |
| **Instituição** | Pontifícia Universidade Católica de Goiás (PUC Goiás), Goiânia, GO |
| **Laboratório** | Laboratório de Computação Científica (LCC) |
| **Fomento** | PROPE/PUC Goiás (Edital 05/2025) |

> **Status: resultado exploratório.** O pipeline é tecnicamente viável, mas os dados atuais
> (3 imagens resistentes, todas ATCC; 16 imagens sensíveis, todas clínicas) confundem
> *resistência* com *procedência*. Não é possível afirmar que o modelo detecta resistência.
> Detalhes e pendências em [docs/VALIDACAO.md](docs/VALIDACAO.md).

## Motivação

A resistência antimicrobiana esteve associada a cerca de 4,95 milhões de mortes em 2019
(Antimicrobial Resistance Collaborators, 2022), e o diagnóstico convencional (cultura +
antibiograma) leva de 48 a 72 h. A imagem hiperespectral SWIR registra um espectro de 256 bandas
(1000–2500 nm) por pixel, sensível à composição química (ligações O–H, C–H, N–H). A hipótese
testada aqui, **não um fato estabelecido**, é que esse espectro carrega sinal do fenótipo de
resistência e que o XGBoost (Chen; Guestrin, 2016) consegue explorá-lo.

## Pipeline

```
ROI (segmentação da colônia)
 ├─ raw   : bandas brutas
 ├─ msc   : Multiplicative Scatter Correction            (Geladi et al., 1985)
 ├─ log   : Log(1/R)
 └─ deriv : 1ª derivada Savitzky-Golay, janela 15, grau 2  (Savitzky; Golay, 1964)
      └─ atributos por pixel (spec_ + idx_ = 23 após remover desc_* e *_area_norm)
      └─ 4 datasets completos em features_selected/ (sem seleção)
split por imagem 70/30 (estratificado por classe)
 → seleção MI / XGB-FI (só no treino, refeita a cada fold)
 → StandardScaler → PCA (95% var.) → XGBoost
 → Grid Search (324 combinações) com StratifiedGroupKFold por imagem
 → avaliação única no teste externo (pixel e imagem)
```

Decisão metodológica central: **o grupo de validação é a imagem**, nunca o pixel. Pixels da mesma
colônia são quase idênticos; separá-los entre treino e validação seria pseudo-replicação
(Hurlbert, 1984) e inflaria o desempenho. A unidade biológica independente é a cepa/isolado.

## Uso

```bash
pip install -r requirements.txt      # requirements-dev.txt: notebooks e testes
python run.py check                  # verifica ambiente / GPU
python run.py all                    # etl -> split -> train -> evaluate
python -m pytest tests               # testes das garantias de validação
```

Etapas isoladas: `python run.py etl | split | train | evaluate`. Dados HSI em
`dataSet/hsi_original/` (não versionados). Variáveis de ambiente: `IC2025_BASE_DIR`,
`IC2025_GRID=fast`, `IC2025_MAX_PIXELS_PER_IMG`, `IC2025_SELECAO_MAX_AMOSTRAS`,
`IC2025_FORCE_CPU=1`.

## Estrutura

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
| `tests/` | Testes das garantias de validação |
| `docs/VALIDACAO.md` | Conferência código × slides × guia de estudo |

## Limitações

- Todas as amostras resistentes são cepas ATCC e todas as sensíveis são isolados clínicos:
  resistência e procedência estão confundidas.
- O n efetivo da classe resistente é 3 cepas, apesar de mais de 1 milhão de pixels.
- O rótulo vem da procedência, não de antibiograma por isolado.
- A PCA antes do XGBoost impede identificar quais bandas pesam no resultado.

## Referências

- ANTIMICROBIAL RESISTANCE COLLABORATORS. Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis. *The Lancet*, v. 399, n. 10325, p. 629–655, 2022.
- CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. In: ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 22., 2016, San Francisco. *Proceedings* [...]. New York: ACM, 2016. p. 785–794.
- GELADI, P.; MACDOUGALL, D.; MARTENS, H. Linearization and scatter-correction for near-infrared reflectance spectra of meat. *Applied Spectroscopy*, v. 39, n. 3, p. 491–500, 1985.
- HURLBERT, S. H. Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, v. 54, n. 2, p. 187–211, 1984.
- PLAZA, A. et al. Recent advances in techniques for hyperspectral image processing. *Remote Sensing of Environment*, v. 113, supl. 1, p. S110–S122, 2009.
- SAVITZKY, A.; GOLAY, M. J. E. Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, v. 36, n. 8, p. 1627–1639, 1964.

## Como citar

> FARIA, J. F.; COELHO, C. J. *Classificação espectral de bactérias multirresistentes com Gradient
> Boosting aplicado a imagens hiperespectrais SWIR*. Goiânia: Laboratório de Computação
> Científica, PUC Goiás, 2025. Disponível em: <https://github.com/jeffersonfelex/ic-2025>.

## Agradecimentos

Ao **Laboratório de Computação Científica (LCC) da PUC Goiás**, pela infraestrutura, pelo
ambiente de pesquisa e pelo apoio ao longo deste trabalho; ao orientador, Prof. Clarimar José
Coelho; à **PROPE/PUC Goiás** (Edital 05/2025); e à equipe que cedeu os isolados bacterianos.

## Autoria

Desenvolvido por **Jefferson Felex de Faria** ([@jeffersonfelex](https://github.com/jeffersonfelex)),
sob orientação de Clarimar José Coelho, no LCC/PUC Goiás. O código foi revisado e reorganizado com
auxílio do Claude Code (Anthropic); a autoria científica e a responsabilidade pelos resultados são
do autor.
