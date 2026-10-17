"""
Treinamento dos 8 pipelines (4 branches x 2 selecoes de features).

Para cada pipeline:
  StandardScaler -> PCA (95% variancia) -> XGBoost
  Grid Search (324 combinacoes) com StratifiedGroupKFold agrupando por imagem.

O numero de folds e adaptativo: com apenas 3 isolados resistentes (ATCC), e
impossivel um k=5 estratificado por grupo. Usa-se
    n_splits = min(5, n_grupos_da_classe_minoritaria_no_treino)
o que, na pratica, equivale a um Leave-One-Resistant-Group-Out.

Saidas em ml/results/:
  comparacao_pipelines.csv, fold_scores_<nome>.csv, grid_results_<nome>.csv,
  pca_importance_<nome>.csv, boxplot_folds_<nome>.png, boxplot_comparativo_pipelines.png
Modelos em ml/models/xgb_<nome>.json
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import scipy.stats as stats

from datetime import timedelta

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, average_precision_score,
)
from xgboost import XGBClassifier

# ==============================================================================
# CONFIGURACAO
# ==============================================================================

BASE_DIR = os.getenv(
    "IC2025_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")
DATA_DIR     = os.path.join(BASE_DIR, "ml", "data")
MODEL_DIR    = os.path.join(BASE_DIR, "ml", "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "ml", "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PIPELINES = [
    ("msc", "mi"), ("msc", "xgb"),
    ("log", "mi"), ("log", "xgb"),
    ("deriv", "mi"), ("deriv", "xgb"),
    ("raw", "mi"), ("raw", "xgb"),
]

PCA_N_COMPONENTS = 0.95
MAX_SPLITS = 5

# --- Aceleradores (variaveis de ambiente) -------------------------------------
# IC2025_GRID = full (padrao, 324 combinacoes) | fast (12 combinacoes)
# IC2025_MAX_PIXELS_PER_IMG = N  -> amostra ate N pixels por imagem no TREINO
#     (0 = usar todos). Pixels da mesma colonia sao pseudo-replicas espaciais;
#     limitar a ~20-30 mil por imagem quase nao altera o sinal e corta MUITO o tempo.
GRID_MODE = os.getenv("IC2025_GRID", "full").strip().lower()
MAX_PIXELS_PER_IMG = int(os.getenv("IC2025_MAX_PIXELS_PER_IMG", "0"))

PARAM_GRID_FULL = {
    "xgb__n_estimators":     [100, 200, 300],
    "xgb__max_depth":        [3, 5, 7],
    "xgb__learning_rate":    [0.01, 0.1, 0.3],
    "xgb__subsample":        [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
    "xgb__gamma":            [0, 0.1, 0.3],
}
PARAM_GRID_FAST = {
    "xgb__n_estimators":     [200, 300],
    "xgb__max_depth":        [3, 5],
    "xgb__learning_rate":    [0.05, 0.1, 0.3],
    "xgb__subsample":        [0.8],
    "xgb__colsample_bytree": [0.8],
    "xgb__gamma":            [0],
}
PARAM_GRID = PARAM_GRID_FAST if GRID_MODE == "fast" else PARAM_GRID_FULL

SELECAO_LABEL = {"mi": "Mutual Information", "xgb": "XGBoost Importance"}

TOTAL_COMBINACOES = (
    len(PARAM_GRID["xgb__n_estimators"]) * len(PARAM_GRID["xgb__max_depth"]) *
    len(PARAM_GRID["xgb__learning_rate"]) * len(PARAM_GRID["xgb__subsample"]) *
    len(PARAM_GRID["xgb__colsample_bytree"]) * len(PARAM_GRID["xgb__gamma"])
)


def subamostrar_por_grupo(X, y, grupos, max_por_grupo: int, seed: int = 42):
    """Mantem no maximo `max_por_grupo` pixels por imagem (amostragem sem reposicao)."""
    if not max_por_grupo or max_por_grupo <= 0:
        return X, y, grupos
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(grupos):
        idx = np.where(grupos == g)[0]
        if len(idx) > max_por_grupo:
            idx = rng.choice(idx, size=max_por_grupo, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], grupos[keep]


def formatar_tempo(segundos: float) -> str:
    return str(timedelta(seconds=int(segundos)))


def detectar_dispositivo() -> str:
    if os.getenv("IC2025_FORCE_CPU", "").strip() == "1":
        print("IC2025_FORCE_CPU=1 — treinando com CPU")
        return "cpu"
    try:
        rng = np.random.default_rng(0)
        Xt = rng.random((256, 6))
        yt = (Xt[:, 0] > 0.5).astype(int)
        test = XGBClassifier(device="cuda", tree_method="hist", n_estimators=8, max_depth=3)
        test.fit(Xt, yt)
        test.predict(Xt)
        print("GPU detectada — treinando com CUDA")
        return "cuda"
    except Exception as e:
        print(f"GPU nao disponivel — treinando com CPU  ({type(e).__name__}: {str(e)[:180]})")
        return "cpu"


def ic95(scores: list) -> tuple:
    """IC95% (t de Student) da media por fold, limitado a [0, 1].

    Com poucos folds (n < 3) e folds nao independentes (pixels da mesma
    cepa), o intervalo t e instavel (com n=2 chega a [-1.8, 2.2]); nesse caso
    devolve NaN e a leitura correta e olhar os scores por fold.
    """
    scores = np.asarray([v for v in scores if v == v], dtype=float)
    n = len(scores)
    if n < 3:
        return (float("nan"), float("nan"))
    se = stats.sem(scores)
    h = se * stats.t.ppf(0.975, df=n - 1)
    m = np.mean(scores)
    return (round(float(max(0.0, m - h)), 4), round(float(min(1.0, m + h)), 4))


def _scorer(fn, usar_proba: bool = False):
    """Scorer que devolve NaN (metrica indefinida) se a validacao nao tem as 2 classes.

    O scikit-learn registra 0 para recall/AUPRC num fold sem positivos; esse 0
    nao e desempenho ruim, e ausencia de dado, e puxaria a media para baixo.
    """
    def score(est, X, y):
        y = np.asarray(y)
        if y.sum() == 0 or y.sum() == len(y):
            return float("nan")
        saida = est.predict_proba(X)[:, 1] if usar_proba else est.predict(X)
        return float(fn(y, saida))
    return score


SCORING_CV = {
    "recall":            _scorer(lambda y, p: recall_score(y, p, zero_division=0)),
    "f1":                _scorer(lambda y, p: f1_score(y, p, zero_division=0)),
    "precision":         _scorer(lambda y, p: precision_score(y, p, zero_division=0)),
    "roc_auc":           _scorer(roc_auc_score, usar_proba=True),
    "average_precision": _scorer(average_precision_score, usar_proba=True),
}


def n_splits_para(y: np.ndarray, grupos: np.ndarray) -> int:
    """min(5, nº de grupos distintos da classe menos representada)."""
    counts = []
    for classe in np.unique(y):
        g = np.unique(grupos[y == classe])
        counts.append(len(g))
    return int(max(1, min(MAX_SPLITS, min(counts))))


def _nan_ou(fn, scores) -> float:
    if np.all(np.isnan(scores)):
        return float("nan")
    return round(float(fn(scores)), 4)


def analisar_folds(cv_results: dict, nome: str) -> dict:
    metricas_fold = {}
    for chave in [k for k in cv_results if k.startswith("test_")]:
        nome_metrica = chave.replace("test_", "")
        scores = [float(s) for s in cv_results[chave]]
        ci = ic95(scores)
        metricas_fold[nome_metrica] = {
            "scores_por_fold": [round(s, 4) for s in scores],
            "media":          _nan_ou(np.nanmean, scores),
            "std":            _nan_ou(np.nanstd, scores),
            "ic95_lower":     ci[0],
            "ic95_upper":     ci[1],
            "folds_indefinidos": [i for i, v in enumerate(scores) if v != v],
        }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Per-Fold Distribution — {nome.upper()}", fontsize=12)
    for ax, metrica_key, titulo in zip(
        axes, ["recall", "f1"], ["Recall per Fold (primary)", "F1 per Fold"]
    ):
        if metrica_key in metricas_fold:
            scores = [v for v in metricas_fold[metrica_key]["scores_por_fold"] if v == v]
            if not scores:
                continue
            ax.boxplot(scores, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.7))
            ax.scatter([1] * len(scores), scores, color="red", zorder=5, s=40)
            ax.set_title(titulo, fontsize=10)
            ax.set_ylabel("Score")
            ax.set_xticks([1]); ax.set_xticklabels([nome.upper()])
            ax.set_ylim(-0.05, 1.05)
            m = float(np.mean(scores))
            ax.axhline(y=m, color="green", linestyle="--", linewidth=1, label=f"Mean={m:.3f}")
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_folds_{nome}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return metricas_fold


def treinar(branch: str, selecao: str, idx: int, total: int, device: str) -> dict:
    nome = f"{branch}_{selecao}"
    print("\n" + "=" * 70)
    print(f"[{idx}/{total}] Pipeline: {branch.upper()} + {SELECAO_LABEL[selecao]}")
    print("=" * 70)

    # Preferir o conjunto de TREINO externo (split_dataset.py); cair para o dataset
    # completo apenas se o split ainda nao foi executado.
    train_path   = os.path.join(DATA_DIR, f"train_{nome}.csv")
    dataset_path = train_path if os.path.exists(train_path) else os.path.join(FEATURES_DIR, f"dataset_{nome}.csv")
    if not os.path.exists(dataset_path):
        print(f"  Dataset nao encontrado: {dataset_path}")
        return {"pipeline": nome, "status": "dataset nao encontrado"}
    print(f"  Fonte: {os.path.relpath(dataset_path, BASE_DIR)}")

    df = pd.read_csv(dataset_path)

    colunas_vazadas = [
        c for c in df.columns
        if c.startswith("desc_") or c.endswith("_area_norm") or c in ("sample_id", "label")
    ]
    feature_cols = [c for c in df.columns if c not in colunas_vazadas]

    for col in ("sample_id", "label"):
        if col not in df.columns:
            print(f"  ERRO: coluna '{col}' ausente em {os.path.basename(dataset_path)}. Pulando.")
            return {"pipeline": nome, "status": f"sem coluna {col}"}

    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    grupos = df["sample_id"].values

    classes = np.unique(y)
    if len(classes) < 2:
        print(f"  ERRO: dataset de treino tem apenas a classe {classes.tolist()} — "
              f"split provavelmente corrompido. Rode 'python run.py split' de novo. Pulando.")
        return {"pipeline": nome, "status": "treino monoclasse"}

    if MAX_PIXELS_PER_IMG:
        n0 = len(y)
        X, y, grupos = subamostrar_por_grupo(X, y, grupos, MAX_PIXELS_PER_IMG)
        print(f"  subamostragem: {n0} -> {len(y)} pixels (max {MAX_PIXELS_PER_IMG}/imagem)")

    n_grupos = len(np.unique(grupos))
    n_splits = n_splits_para(y, grupos)
    total_fits = TOTAL_COMBINACOES * n_splits

    print(f"  {X.shape[0]} pixels x {X.shape[1]} features (anti-leakage aplicado)")
    print(f"  Imagens (grupos): {n_grupos} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  n_splits = {n_splits}  (adaptativo; {TOTAL_COMBINACOES} combinacoes x {n_splits} folds = {total_fits} fits)")

    if n_splits < 2:
        print("  ERRO: menos de 2 grupos na classe minoritaria — CV por grupo impossivel. Pulando.")
        return {"pipeline": nome, "status": "grupos insuficientes"}

    # n_jobs=1 no XGBoost: o paralelismo fica a cargo do GridSearchCV (loky),
    # evitando oversubscription de threads.
    xgb_base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        device=device, random_state=42, n_jobs=1,
    )
    pipeline_sklearn = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=PCA_N_COMPONENTS, random_state=42)),
        ("xgb",    xgb_base),
    ])

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_jobs = 1 if device == "cuda" else -1

    scoring = SCORING_CV

    print(f"  Grid Search em andamento ({total_fits} fits, n_jobs={n_jobs})...", flush=True)
    grid = GridSearchCV(
        estimator=pipeline_sklearn, param_grid=PARAM_GRID,
        scoring="recall", refit="recall", cv=cv, n_jobs=n_jobs,
        verbose=1, return_train_score=False,
    )
    t_inicio = time.time()
    grid.fit(X, y, groups=grupos)

    t_decorrido = time.time() - t_inicio
    best_pipeline = grid.best_estimator_
    print(f"  Concluido em {formatar_tempo(t_decorrido)}")
    print(f"  Melhores hiperparametros: {grid.best_params_}")

    cv_detalhado = cross_validate(
        best_pipeline, X, y, groups=grupos, cv=cv,
        scoring=scoring, return_train_score=False, n_jobs=n_jobs,
    )
    mf = analisar_folds(cv_detalhado, nome)

    cv_recall_mean = mf.get("recall", {}).get("media", float("nan"))
    cv_recall_std  = mf.get("recall", {}).get("std", float("nan"))
    cv_recall_ic95 = (mf.get("recall", {}).get("ic95_lower", float("nan")),
                      mf.get("recall", {}).get("ic95_upper", float("nan")))
    cv_f1_mean    = mf.get("f1", {}).get("media", float("nan"))
    cv_f1_std     = mf.get("f1", {}).get("std", float("nan"))
    cv_prec_mean  = mf.get("precision", {}).get("media", float("nan"))
    cv_rocauc_mean = mf.get("roc_auc", {}).get("media", float("nan"))
    cv_auprc_mean = mf.get("average_precision", {}).get("media", float("nan"))
    cv_auprc_std  = mf.get("average_precision", {}).get("std", float("nan"))
    folds_indefinidos = mf.get("recall", {}).get("folds_indefinidos", [])

    best_pipeline.set_params(xgb__device="cpu")
    y_pred  = best_pipeline.predict(X)
    y_proba = best_pipeline.predict_proba(X)[:, 1]
    best_pipeline.set_params(xgb__device=device)

    insample_recall = round(recall_score(y, y_pred, zero_division=0), 4)
    insample_f1     = round(f1_score(y, y_pred, zero_division=0), 4)
    insample_auprc  = round(average_precision_score(y, y_proba), 4)
    n_pca = best_pipeline.named_steps["pca"].n_components_

    metricas = {
        "pipeline": nome, "branch": branch, "selecao": SELECAO_LABEL[selecao],
        "device": device, "n_splits": n_splits,
        "cv_recall_mean": cv_recall_mean, "cv_recall_std": cv_recall_std,
        "cv_recall_ic95_lower": cv_recall_ic95[0], "cv_recall_ic95_upper": cv_recall_ic95[1],
        "cv_f1_mean": cv_f1_mean, "cv_f1_std": cv_f1_std,
        "cv_precision_mean": cv_prec_mean,
        "cv_roc_auc_mean": cv_rocauc_mean,
        "cv_auprc_mean": cv_auprc_mean, "cv_auprc_std": cv_auprc_std,
        "folds_indefinidos": str(folds_indefinidos),
        "insample_recall": insample_recall, "insample_f1": insample_f1,
        "insample_auprc": insample_auprc,
        "insample_roc_auc": round(roc_auc_score(y, y_proba), 4),
        "n_features_originais": X.shape[1], "n_componentes_pca": n_pca,
        "n_grupos_imagens": n_grupos, "tempo_s": round(t_decorrido, 1),
        **{f"param_{k.replace('xgb__', '')}": v for k, v in grid.best_params_.items()},
    }

    gap = round(insample_recall - cv_recall_mean, 4) if cv_recall_mean == cv_recall_mean else "N/A"
    print(f"\n  MetriCAS [{nome.upper()}]  PCA: {X.shape[1]} -> {n_pca} componentes")
    print(f"  cv_recall : {cv_recall_mean} +/- {cv_recall_std}  IC95%=[{cv_recall_ic95[0]}, {cv_recall_ic95[1]}]")
    print(f"  cv_f1     : {cv_f1_mean} +/- {cv_f1_std}")
    print(f"  cv_auprc  : {cv_auprc_mean} +/- {cv_auprc_std}   cv_roc_auc: {cv_rocauc_mean}")
    if folds_indefinidos:
        print(f"  ATENCAO: folds sem imagem resistente (metrica indefinida, ignorados): {folds_indefinidos}")
    print(f"  in-sample recall {insample_recall} | gap overfitting (IS - CV): {gap}")
    print(f"  Matriz de confusao (in-sample):\n{confusion_matrix(y, y_pred)}")

    best_pipeline.named_steps["xgb"].get_booster().save_model(
        os.path.join(MODEL_DIR, f"xgb_{nome}.json")
    )
    # Pipeline completo (scaler + PCA + XGBoost) exatamente como foi treinado:
    # evaluate.py o carrega, em vez de reajustar scaler/PCA por conta propria.
    best_pipeline.set_params(xgb__device="cpu")
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, f"pipeline_{nome}.joblib"))
    pd.DataFrame(grid.cv_results_).to_csv(
        os.path.join(RESULTS_DIR, f"grid_results_{nome}.csv"), index=False
    )
    fold_rows = [
        {"pipeline": nome, "metrica": mk, "fold": fi, "score": s}
        for mk, d in mf.items()
        for fi, s in enumerate(d["scores_por_fold"])
    ]
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(RESULTS_DIR, f"fold_scores_{nome}.csv"), index=False
    )
    pca_imp = best_pipeline.named_steps["xgb"].feature_importances_
    pd.DataFrame({
        "componente_pca": [f"PC{i+1}" for i in range(len(pca_imp))],
        "importance": pca_imp,
    }).sort_values("importance", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, f"pca_importance_{nome}.csv"), index=False
    )
    print(f"  Modelo salvo: xgb_{nome}.json | pipeline_{nome}.joblib")
    return metricas


def _gerar_boxplot_comparativo(resultados: list) -> None:
    fold_files = [os.path.join(RESULTS_DIR, f"fold_scores_{r['pipeline']}.csv") for r in resultados]
    dfs = [pd.read_csv(f) for f in fold_files if os.path.exists(f)]
    if not dfs:
        return
    df_all = pd.concat(dfs, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pipeline Comparison — CV per Fold", fontsize=13)
    for ax, metrica, titulo in zip(
        axes, ["recall", "average_precision"],
        ["Recall per Fold (primary)", "AUPRC per Fold"]
    ):
        sub = df_all[df_all["metrica"] == metrica]
        if sub.empty:
            continue
        pipes = list(sub["pipeline"].unique())
        ax.boxplot([sub[sub["pipeline"] == p]["score"].values for p in pipes],
                   patch_artist=True)
        ax.set_title(titulo, fontsize=11); ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(1, len(pipes) + 1))
        ax.set_xticklabels(pipes, rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "boxplot_comparativo_pipelines.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot comparativo salvo em: {path}")


def main():
    total = len(PIPELINES)
    t_global = time.time()
    print("Treinamento — 8 pipelines (4 branches x 2 selecoes)")
    print(f"Pipeline: StandardScaler -> PCA (95% var) -> XGBoost")
    print(f"Grid: {GRID_MODE.upper()} ({TOTAL_COMBINACOES} combinacoes) | "
          f"max pixels/imagem no treino: {MAX_PIXELS_PER_IMG or 'todos'}")
    device = detectar_dispositivo()

    resultados = []
    for idx, (branch, selecao) in enumerate(PIPELINES, start=1):
        try:
            resultados.append(treinar(branch, selecao, idx, total, device))
        except Exception as e:
            print(f"  ERRO no pipeline {branch}_{selecao}: {type(e).__name__}: {str(e)[:200]}")
            resultados.append({"pipeline": f"{branch}_{selecao}", "status": f"erro: {type(e).__name__}"})
        print(f"  Tempo total ate agora: {formatar_tempo(time.time() - t_global)}")

    validos = [r for r in resultados if "cv_recall_mean" in r]
    if not validos:
        print("\nNenhum pipeline concluido com sucesso.")
        return

    cols = [
        "pipeline", "branch", "selecao", "n_splits",
        "cv_recall_mean", "cv_recall_std", "cv_recall_ic95_lower", "cv_recall_ic95_upper",
        "cv_f1_mean", "cv_f1_std", "cv_precision_mean", "cv_roc_auc_mean",
        "cv_auprc_mean", "cv_auprc_std", "folds_indefinidos",
        "insample_recall", "insample_f1",
        "n_features_originais", "n_componentes_pca", "n_grupos_imagens", "tempo_s",
    ]
    df_result = pd.DataFrame(validos)[cols].sort_values("cv_recall_mean", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 70)
    print("COMPARACAO FINAL")
    print("=" * 70)
    print(df_result.to_string(index=False))

    if any(s > 0.20 for s in df_result["cv_recall_std"].values):
        print("\n  ATENCAO: std > 0.20 em algum pipeline — ranking NAO e estatisticamente robusto.")

    out = os.path.join(RESULTS_DIR, "comparacao_pipelines.csv")
    df_result.to_csv(out, index=False)
    print(f"\nRelatorio salvo em: {out}")
    print(f"Tempo total: {formatar_tempo(time.time() - t_global)}")
    _gerar_boxplot_comparativo(validos)


if __name__ == "__main__":
    main()
