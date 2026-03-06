import os
import time
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

BASE_DIR     = r"C:\Users\jefferson.faria\Documents\GitHub\ic-2025"
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")
MODEL_DIR    = os.path.join(BASE_DIR, "ml", "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "ml", "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PIPELINES = [
    ("msc",   "mi"),
    ("msc",   "xgb"),
    ("log",   "mi"),
    ("log",   "xgb"),
    ("deriv", "mi"),
    ("deriv", "xgb"),
]

PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.1, 0.3],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma":            [0, 0.1, 0.3],
}

SELECAO_LABEL = {
    "mi":  "Mutual Information",
    "xgb": "XGBoost Importance",
}

TOTAL_COMBINACOES = (
    len(PARAM_GRID["n_estimators"]) *
    len(PARAM_GRID["max_depth"]) *
    len(PARAM_GRID["learning_rate"]) *
    len(PARAM_GRID["subsample"]) *
    len(PARAM_GRID["colsample_bytree"]) *
    len(PARAM_GRID["gamma"])
)  # 324 combinações × 5 folds = 1620 treinamentos por pipeline


def formatar_tempo(segundos: float) -> str:
    return str(timedelta(seconds=int(segundos)))


# ==============================================================================
# TREINAMENTO DE UM PIPELINE
# ==============================================================================

def treinar(branch: str, selecao: str, idx: int, total: int) -> dict:
    nome = f"{branch}_{selecao}"

    print("\n" + "=" * 70)
    print(f"🚀 [{idx}/{total}] Pipeline: {branch.upper()} + {SELECAO_LABEL[selecao]}")
    print(f"   Grid: {TOTAL_COMBINACOES} combinações × 5 folds = {TOTAL_COMBINACOES * 5} treinamentos")
    print("=" * 70)

    dataset_path = os.path.join(FEATURES_DIR, f"dataset_{nome}.csv")
    if not os.path.exists(dataset_path):
        print(f"  ❌ Dataset não encontrado: {dataset_path}")
        return {"pipeline": nome, "status": "dataset não encontrado"}

    df = pd.read_csv(dataset_path)
    feature_cols = [c for c in df.columns if c not in ("sample_id", "label")]
    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    print(f"  📥 {X.shape[0]} amostras × {X.shape[1]} features")
    print(f"     Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------------------------------------------------------------------
    # GRID SEARCH
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,  # silenciado — usamos nosso próprio progresso
    )

    print(f"  ⏳ Iniciando Grid Search...")
    t_inicio = time.time()

    grid.fit(X, y)

    t_fim      = time.time()
    t_decorrido = t_fim - t_inicio

    best_model = grid.best_estimator_
    print(f"  ✅ Grid Search concluído em {formatar_tempo(t_decorrido)}")
    print(f"  ✅ Melhores hiperparâmetros: {grid.best_params_}")

    # ------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------
    y_pred  = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]

    metricas = {
        "pipeline":   nome,
        "branch":     branch,
        "selecao":    SELECAO_LABEL[selecao],
        "acuracia":   round(accuracy_score(y, y_pred), 4),
        "f1":         round(f1_score(y, y_pred, zero_division=0), 4),
        "precisao":   round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y, y_pred, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y, y_proba), 4),
        "n_features": X.shape[1],
        "tempo_s":    round(t_decorrido, 1),
        **{f"param_{k}": v for k, v in grid.best_params_.items()},
    }

    print(f"\n  📊 Métricas [{nome.upper()}]")
    for k in ("acuracia", "f1", "precisao", "recall", "roc_auc"):
        print(f"     {k:<10}: {metricas[k]}")
    print(f"     Matriz de confusão:\n{confusion_matrix(y, y_pred)}")

    # ------------------------------------------------------------------
    # SALVAR
    # ------------------------------------------------------------------
    best_model.save_model(os.path.join(MODEL_DIR, f"xgb_{nome}.json"))

    pd.DataFrame(grid.cv_results_).to_csv(
        os.path.join(RESULTS_DIR, f"grid_results_{nome}.csv"), index=False
    )

    pd.DataFrame({
        "feature":    feature_cols,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, f"feature_importance_{nome}.csv"), index=False
    )

    print(f"  💾 Modelo salvo: xgb_{nome}.json")
    return metricas


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    total     = len(PIPELINES)
    t_global  = time.time()
    tempos    = []

    print("🧬 Treinamento — 6 pipelines independentes")
    print("   MSC   × MI  |  MSC   × XGBoost Importance")
    print("   Log   × MI  |  Log   × XGBoost Importance")
    print("   Deriv × MI  |  Deriv × XGBoost Importance")
    print(f"\n   Total de combinações por pipeline : {TOTAL_COMBINACOES} × 5 folds")
    print(f"   Total de treinamentos estimados   : {TOTAL_COMBINACOES * 5 * total}\n")

    resultados = []
    for idx, (branch, selecao) in enumerate(PIPELINES, start=1):
        t0      = time.time()
        result  = treinar(branch, selecao, idx, total)
        elapsed = time.time() - t0
        tempos.append(elapsed)
        resultados.append(result)

        # Estimativa de tempo restante
        media   = sum(tempos) / len(tempos)
        restante = media * (total - idx)
        print(f"\n  ⏱️  Tempo deste pipeline : {formatar_tempo(elapsed)}")
        print(f"  ⏱️  Tempo total até agora: {formatar_tempo(time.time() - t_global)}")
        if idx < total:
            print(f"  ⏱️  Estimativa restante  : {formatar_tempo(restante)} ({total - idx} pipelines)")

    # ------------------------------------------------------------------
    # RELATÓRIO FINAL
    # ------------------------------------------------------------------
    t_total = time.time() - t_global

    print("\n" + "=" * 70)
    print("📊 COMPARAÇÃO FINAL — 6 pipelines")
    print("=" * 70)

    cols_resumo = ["pipeline", "branch", "selecao", "acuracia", "f1",
                   "precisao", "recall", "roc_auc", "n_features", "tempo_s"]
    df_result = (
        pd.DataFrame(resultados)[cols_resumo]
        .sort_values("acuracia", ascending=False)
        .reset_index(drop=True)
    )
    print(df_result.to_string(index=False))

    melhor = df_result.iloc[0]
    print(f"\n🏆 Melhor pipeline  : {melhor['pipeline'].upper()}")
    print(f"   Transformação    : {melhor['branch'].upper()}")
    print(f"   Seleção          : {melhor['selecao']}")
    print(f"   Acurácia         : {melhor['acuracia']}")
    print(f"   F1-Score         : {melhor['f1']}")
    print(f"   ROC-AUC          : {melhor['roc_auc']}")
    print(f"\n⏱️  Tempo total      : {formatar_tempo(t_total)}")

    relatorio = os.path.join(RESULTS_DIR, "comparacao_pipelines.csv")
    df_result.to_csv(relatorio, index=False)
    print(f"✅ Relatório salvo em: {relatorio}")


if __name__ == "__main__":
    main()