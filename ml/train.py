import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

from datetime import timedelta
from tqdm import tqdm
from joblib import parallel_backend

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    average_precision_score
)
from xgboost import XGBClassifier

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

BASE_DIR     = r"C:\Users\Jefferson\Documents\GitHub\ic-2025"
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")
MODEL_DIR    = os.path.join(BASE_DIR, "ml", "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "ml", "results")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PIPELINES = [
    ("msc",   "mi"),
    ("msc",   "xgb"),
    ("log",   "mi"),
    ("log",   "xgb"),
    ("deriv", "mi"),
    ("deriv", "xgb"),
    ("raw",   "mi"),
    ("raw",   "xgb"),
]

# PCA: número de componentes (None = manter 95% da variância)
PCA_N_COMPONENTS = 0.95

PARAM_GRID = {
    "xgb__n_estimators":     [100, 200, 300],
    "xgb__max_depth":        [3, 5, 7],
    "xgb__learning_rate":    [0.01, 0.1, 0.3],
    "xgb__subsample":        [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
    "xgb__gamma":            [0, 0.1, 0.3],
}

SELECAO_LABEL = {
    "mi":  "Mutual Information",
    "xgb": "XGBoost Importance",
}

TOTAL_COMBINACOES = (
    len(PARAM_GRID["xgb__n_estimators"]) *
    len(PARAM_GRID["xgb__max_depth"]) *
    len(PARAM_GRID["xgb__learning_rate"]) *
    len(PARAM_GRID["xgb__subsample"]) *
    len(PARAM_GRID["xgb__colsample_bytree"]) *
    len(PARAM_GRID["xgb__gamma"])
)
TOTAL_FITS = TOTAL_COMBINACOES * 5  # 5 folds


def formatar_tempo(segundos: float) -> str:
    return str(timedelta(seconds=int(segundos)))


def detectar_dispositivo() -> str:
    try:
        test = XGBClassifier(device="cuda", n_estimators=1)
        test.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        print("GPU detectada — treinando com CUDA")
        return "cuda"
    except Exception:
        print("GPU não disponível — treinando com CPU")
        return "cpu"


def ic95(scores: list) -> tuple:
    """Calcula intervalo de confiança 95% via t-distribution."""
    n = len(scores)
    if n < 2:
        return (float("nan"), float("nan"))
    se = stats.sem(scores)
    h = se * stats.t.ppf(0.975, df=n - 1)
    m = np.mean(scores)
    return (round(m - h, 4), round(m + h, 4))


# ==============================================================================
# WRAPPER COM PROGRESSO
# ==============================================================================

class XGBComProgresso(XGBClassifier, ClassifierMixin):
    """Wrapper que atualiza a barra de progresso a cada fit() chamado pelo GridSearchCV."""
    _estimator_type = "classifier"
    pbar: tqdm = None

    def fit(self, X, y, **kwargs):
        result = super().fit(X, y, **kwargs)
        if XGBComProgresso.pbar is not None:
            XGBComProgresso.pbar.update(1)
        return result


# ==============================================================================
# ANÁLISE ROBUSTA POR FOLD
# ==============================================================================

def analisar_folds(cv_results: dict, nome: str) -> dict:
    """
    Extrai métricas por fold, calcula IC95% e identifica folds problemáticos.
    Gera boxplot por fold salvo em RESULTS_DIR.
    """
    metricas_fold = {}
    fold_metricas_disponiveis = [k for k in cv_results if k.startswith("test_")]

    for chave in fold_metricas_disponiveis:
        nome_metrica = chave.replace("test_", "")
        scores = list(cv_results[chave])
        media  = round(float(np.mean(scores)), 4)
        std    = round(float(np.std(scores)),  4)
        ci     = ic95(scores)
        folds_zero = [i for i, s in enumerate(scores) if s == 0.0]

        metricas_fold[nome_metrica] = {
            "scores_por_fold": [round(s, 4) for s in scores],
            "media":           media,
            "std":             std,
            "ic95_lower":      ci[0],
            "ic95_upper":      ci[1],
            "folds_com_zero":  folds_zero,
        }

    # --- Boxplot de recall e f1 por fold ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Distribuição por Fold — {nome.upper()}", fontsize=12)

    for ax, metrica_key, titulo in zip(
        axes,
        ["recall", "f1"],
        ["Recall por Fold (Métrica Primária)", "F1 por Fold"]
    ):
        if metrica_key in metricas_fold:
            scores = metricas_fold[metrica_key]["scores_por_fold"]
            ax.boxplot(scores, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.7))
            ax.scatter([1] * len(scores), scores, color="red", zorder=5, s=40)
            ax.set_title(titulo, fontsize=10)
            ax.set_ylabel("Score")
            ax.set_xticks([1])
            ax.set_xticklabels([nome.upper()])
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=np.mean(scores), color="green", linestyle="--",
                       linewidth=1, label=f"Média={np.mean(scores):.3f}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    boxplot_path = os.path.join(RESULTS_DIR, f"boxplot_folds_{nome}.png")
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return metricas_fold


# ==============================================================================
# TREINAMENTO DE UM PIPELINE
# ==============================================================================

def treinar(branch: str, selecao: str, idx: int, total: int, device: str) -> dict:
    nome = f"{branch}_{selecao}"

    print("\n" + "=" * 70)
    print(f"[{idx}/{total}] Pipeline: {branch.upper()} + {SELECAO_LABEL[selecao]}")
    print(f"   Dispositivo : {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
    print(f"   Combinações : {TOTAL_COMBINACOES} × 5 folds = {TOTAL_FITS} treinamentos")
    print("=" * 70)

    dataset_path = os.path.join(FEATURES_DIR, f"dataset_{nome}.csv")
    if not os.path.exists(dataset_path):
        print(f"  Dataset não encontrado: {dataset_path}")
        return {"pipeline": nome, "status": "dataset não encontrado"}

    df = pd.read_csv(dataset_path)

    # Filtro anti-leakage: remove estatísticas descritivas e áreas
    colunas_vazadas = [
        c for c in df.columns
        if c.startswith("desc_") or c.endswith("_area_norm") or c in ("sample_id", "label")
    ]
    feature_cols = [c for c in df.columns if c not in colunas_vazadas]

    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    grupos = df["sample_id"].values

    n_classes = len(np.unique(y))
    n_grupos  = len(np.unique(grupos))

    print(f"  {X.shape[0]} pixels × {X.shape[1]} features (anti-leakage aplicado)")
    print(f"  Imagens (grupos): {n_grupos} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  NOTA: pixels da mesma imagem compartilham contexto — StratifiedGroupKFold")
    print(f"        garante que imagens inteiras não cruzam entre treino e validação.")

    # ------------------------------------------------------------------
    # PIPELINE: StandardScaler → PCA → XGBoost
    # PCA reduz dimensionalidade e remove redundância espectral antes do modelo.
    # ------------------------------------------------------------------
    xgb_base = XGBComProgresso(
        objective="binary:logistic",
        eval_metric="logloss",
        device=device,
        random_state=42,
    )

    pipeline_sklearn = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=PCA_N_COMPONENTS, random_state=42)),
        ("xgb",    xgb_base),
    ])

    cv     = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    n_jobs = 1 if device == "cuda" else -1

    XGBComProgresso.pbar = tqdm(
        total=TOTAL_FITS,
        desc=f"  Grid Search [{nome}]",
        unit="fit",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    # Scoring principal: recall (sensibilidade = detecção de resistentes é prioridade)
    # AUPRC via average_precision_score
    scoring = {
        "recall":            "recall",
        "f1":                "f1",
        "precision":         "precision",
        "roc_auc":           "roc_auc",
        "average_precision": "average_precision",  # AUPRC
    }

    grid = GridSearchCV(
        estimator=pipeline_sklearn,
        param_grid=PARAM_GRID,
        scoring="recall",          # recall como métrica primária de seleção
        refit="recall",
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=False,
    )

    t_inicio = time.time()

    with parallel_backend("threading" if device == "cuda" else "loky"):
        grid.fit(X, y, groups=grupos)

    XGBComProgresso.pbar.close()
    XGBComProgresso.pbar = None

    t_decorrido = time.time() - t_inicio
    best_pipeline = grid.best_estimator_

    print(f"  Concluído em {formatar_tempo(t_decorrido)}")
    print(f"  Melhores hiperparâmetros: {grid.best_params_}")

    # ------------------------------------------------------------------
    # AVALIAÇÃO POR FOLD (cross_validate com múltiplas métricas)
    # Re-executa CV com o melhor modelo para obter scores detalhados por fold
    # ------------------------------------------------------------------
    cv_detalhado = cross_validate(
        best_pipeline, X, y,
        groups=grupos,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=n_jobs,
    )

    metricas_fold = analisar_folds(cv_detalhado, nome)

    # Métricas out-of-fold (confiáveis)
    cv_recall_mean = metricas_fold.get("recall", {}).get("media", float("nan"))
    cv_recall_std  = metricas_fold.get("recall", {}).get("std",   float("nan"))
    cv_recall_ic95 = (
        metricas_fold.get("recall", {}).get("ic95_lower", float("nan")),
        metricas_fold.get("recall", {}).get("ic95_upper", float("nan")),
    )
    cv_f1_mean     = metricas_fold.get("f1", {}).get("media", float("nan"))
    cv_f1_std      = metricas_fold.get("f1", {}).get("std",   float("nan"))
    cv_auprc_mean  = metricas_fold.get("average_precision", {}).get("media", float("nan"))
    cv_auprc_std   = metricas_fold.get("average_precision", {}).get("std",   float("nan"))

    folds_recall_zero = metricas_fold.get("recall", {}).get("folds_com_zero", [])

    # Métricas in-sample (apenas diagnóstico de overfitting)
    best_pipeline.set_params(xgb__device="cpu")

    y_pred  = best_pipeline.predict(X)
    y_proba = best_pipeline.predict_proba(X)[:, 1]

    best_pipeline.set_params(xgb__device=device)

    insample_recall = round(recall_score(y, y_pred, zero_division=0), 4)
    insample_f1     = round(f1_score(y, y_pred, zero_division=0), 4)
    insample_auprc  = round(average_precision_score(y, y_proba), 4)

    # Determinar número de componentes PCA efetivo
    n_pca_components = best_pipeline.named_steps["pca"].n_components_

    metricas = {
        "pipeline":              nome,
        "branch":                branch,
        "selecao":               SELECAO_LABEL[selecao],
        "device":                device,
        # Out-of-fold (confiáveis)
        "cv_recall_mean":        cv_recall_mean,
        "cv_recall_std":         cv_recall_std,
        "cv_recall_ic95_lower":  cv_recall_ic95[0],
        "cv_recall_ic95_upper":  cv_recall_ic95[1],
        "cv_f1_mean":            cv_f1_mean,
        "cv_f1_std":             cv_f1_std,
        "cv_auprc_mean":         cv_auprc_mean,
        "cv_auprc_std":          cv_auprc_std,
        "folds_recall_zero":     str(folds_recall_zero),
        # In-sample (diagnóstico de overfitting)
        "insample_recall":       insample_recall,
        "insample_f1":           insample_f1,
        "insample_auprc":        insample_auprc,
        "insample_roc_auc":      round(roc_auc_score(y, y_proba), 4),
        # Estrutura
        "n_features_originais":  X.shape[1],
        "n_componentes_pca":     n_pca_components,
        "n_grupos_imagens":      n_grupos,
        "tempo_s":               round(t_decorrido, 1),
        **{f"param_{k.replace('xgb__', '')}": v for k, v in grid.best_params_.items()},
    }

    print(f"\n  MÉTRICAS [{nome.upper()}]")
    print(f"  PCA: {X.shape[1]} features → {n_pca_components} componentes (95% variância)")
    print(f"  ── Cross-Validation out-of-fold (confiável) ──")
    print(f"  cv_recall  : {cv_recall_mean} ± {cv_recall_std}  IC95%=[{cv_recall_ic95[0]}, {cv_recall_ic95[1]}]")
    print(f"  cv_f1      : {cv_f1_mean} ± {cv_f1_std}")
    print(f"  cv_auprc   : {cv_auprc_mean} ± {cv_auprc_std}")
    if folds_recall_zero:
        print(f"  ATENCAO: Folds com recall=0: {folds_recall_zero}  (CV instável)")
    print(f"  ── In-sample / treino completo (diagnóstico de overfitting) ──")
    print(f"  recall (IS): {insample_recall}  |  f1 (IS): {insample_f1}  |  auprc (IS): {insample_auprc}")
    overfitting_gap = round(insample_recall - cv_recall_mean, 4) if cv_recall_mean == cv_recall_mean else "N/A"
    print(f"  Gap overfitting (IS - CV): {overfitting_gap}")
    print(f"  Matriz de confusão (in-sample):\n{confusion_matrix(y, y_pred)}")

    # ------------------------------------------------------------------
    # SALVAR
    # ------------------------------------------------------------------
    best_pipeline.named_steps["xgb"].get_booster().save_model(
        os.path.join(MODEL_DIR, f"xgb_{nome}.json")
    )

    pd.DataFrame(grid.cv_results_).to_csv(
        os.path.join(RESULTS_DIR, f"grid_results_{nome}.csv"), index=False
    )

    # Salvar detalhes dos folds
    fold_rows = []
    for metrica_key, dados in metricas_fold.items():
        for fold_idx, score in enumerate(dados["scores_por_fold"]):
            fold_rows.append({
                "pipeline": nome,
                "metrica":  metrica_key,
                "fold":     fold_idx,
                "score":    score,
            })
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(RESULTS_DIR, f"fold_scores_{nome}.csv"), index=False
    )

    # Feature importance (sobre features originais — antes do PCA)
    # XGBoost após PCA opera sobre componentes; importância refere-se aos PCs
    pca_importances = best_pipeline.named_steps["xgb"].feature_importances_
    pc_names = [f"PC{i+1}" for i in range(len(pca_importances))]
    df_importances = pd.DataFrame({
        "componente_pca": pc_names,
        "importance":     pca_importances,
    }).sort_values("importance", ascending=False)
    df_importances.to_csv(
        os.path.join(RESULTS_DIR, f"pca_importance_{nome}.csv"), index=False
    )

    print(f"  Modelo salvo: xgb_{nome}.json")
    print(f"  Boxplot salvo: boxplot_folds_{nome}.png")
    return metricas


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    total    = len(PIPELINES)
    t_global = time.time()
    tempos   = []

    print("Treinamento — 8 pipelines independentes")
    print("   MSC   × MI  |  MSC   × XGBoost Importance")
    print("   Log   × MI  |  Log   × XGBoost Importance")
    print("   Deriv × MI  |  Deriv × XGBoost Importance")
    print("   Raw   × MI  |  Raw   × XGBoost Importance  (baseline)")
    print(f"\n   Pipeline: StandardScaler → PCA (95% var) → XGBoost")
    print(f"   Combinações por pipeline : {TOTAL_COMBINACOES} × 5 folds = {TOTAL_FITS}")
    print(f"   Métrica primária         : Recall (detecção de resistentes)")
    print(f"   Métrica secundária       : AUPRC (área sob curva precisão-recall)\n")

    device = detectar_dispositivo()

    pbar_global = tqdm(
        total=total,
        desc="Progresso geral",
        unit="pipeline",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} pipelines [{elapsed}<{remaining}]"
    )

    resultados = []
    for idx, (branch, selecao) in enumerate(PIPELINES, start=1):
        t0      = time.time()
        result  = treinar(branch, selecao, idx, total, device)
        elapsed = time.time() - t0
        tempos.append(elapsed)
        resultados.append(result)
        pbar_global.update(1)

        media    = sum(tempos) / len(tempos)
        restante = media * (total - idx)
        print(f"\n  Tempo deste pipeline : {formatar_tempo(elapsed)}")
        print(f"  Tempo total até agora: {formatar_tempo(time.time() - t_global)}")
        if idx < total:
            print(f"  Estimativa restante  : {formatar_tempo(restante)} ({total - idx} pipelines)")

    pbar_global.close()

    # ------------------------------------------------------------------
    # RELATÓRIO FINAL
    # ------------------------------------------------------------------
    t_total = time.time() - t_global

    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL — 6 pipelines")
    print("=" * 70)

    cols_resumo = [
        "pipeline", "branch", "selecao",
        "cv_recall_mean", "cv_recall_std", "cv_recall_ic95_lower", "cv_recall_ic95_upper",
        "cv_f1_mean", "cv_f1_std",
        "cv_auprc_mean", "cv_auprc_std",
        "folds_recall_zero",
        "insample_recall", "insample_f1",
        "n_features_originais", "n_componentes_pca", "n_grupos_imagens",
        "tempo_s",
    ]
    resultados_validos = [r for r in resultados if "cv_recall_mean" in r]

    if not resultados_validos:
        print("\nNenhum pipeline concluído com sucesso.")
        return

    df_result = (
        pd.DataFrame(resultados_validos)[cols_resumo]
        .sort_values("cv_recall_mean", ascending=False)
        .reset_index(drop=True)
    )
    print(df_result.to_string(index=False))

    # Aviso sobre instabilidade — não declarar "melhor pipeline" se CV for instável
    melhor = df_result.iloc[0]
    print(f"\nPipeline com maior cv_recall_mean: {melhor['pipeline'].upper()}")
    print(f"   cv_recall : {melhor['cv_recall_mean']} ± {melhor['cv_recall_std']}")
    print(f"   IC95%     : [{melhor['cv_recall_ic95_lower']}, {melhor['cv_recall_ic95_upper']}]")
    print(f"   cv_auprc  : {melhor['cv_auprc_mean']} ± {melhor['cv_auprc_std']}")

    # Verificar estabilidade do ranking
    recalls = df_result["cv_recall_mean"].values
    stds    = df_result["cv_recall_std"].values
    instavel = any(s > 0.20 for s in stds)
    if instavel:
        print("\n  ATENCAO: Alta variabilidade nos folds (std > 0.20 em algum pipeline).")
        print("  O ranking entre pipelines NÃO é estatisticamente robusto.")
        print("  Interprete os resultados com cautela — poucos grupos (imagens) limitam a CV.")

    print(f"\nTempo total: {formatar_tempo(t_total)}")

    relatorio = os.path.join(RESULTS_DIR, "comparacao_pipelines.csv")
    df_result.to_csv(relatorio, index=False)
    print(f"Relatório salvo em: {relatorio}")

    # Boxplot comparativo de recall entre todos os pipelines
    _gerar_boxplot_comparativo(resultados_validos)


def _gerar_boxplot_comparativo(resultados: list) -> None:
    """Gera boxplot comparativo de recall e AUPRC entre todos os pipelines."""
    fold_files = [
        os.path.join(RESULTS_DIR, f"fold_scores_{r['pipeline']}.csv")
        for r in resultados
    ]

    dfs = []
    for f in fold_files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))

    if not dfs:
        return

    df_all = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparação entre Pipelines — CV por Fold", fontsize=13)

    for ax, metrica, titulo in zip(
        axes,
        ["recall", "average_precision"],
        ["Recall por Fold (Métrica Primária)", "AUPRC por Fold"]
    ):
        sub = df_all[df_all["metrica"] == metrica]
        if sub.empty:
            continue
        pipelines = sub["pipeline"].unique()
        data_plot = [sub[sub["pipeline"] == p]["score"].values for p in pipelines]

        ax.boxplot(data_plot, labels=pipelines, vert=True, patch_artist=True)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "boxplot_comparativo_pipelines.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot comparativo salvo em: {path}")


if __name__ == "__main__":
    main()
