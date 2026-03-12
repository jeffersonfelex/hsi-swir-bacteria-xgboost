import os
import time
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
from joblib import parallel_backend
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

BASE_DIR     = r"C:\Users\Jefferson\Documents\GitHub\ic-2025"
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
)
TOTAL_FITS = TOTAL_COMBINACOES * 5  # 5 folds


def formatar_tempo(segundos: float) -> str:
    return str(timedelta(seconds=int(segundos)))


def detectar_dispositivo() -> str:
    try:
        test = XGBClassifier(device="cuda", n_estimators=1)
        test.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        print("🖥️  GPU detectada — treinando com CUDA")
        return "cuda"
    except Exception:
        print("⚠️  GPU não disponível — treinando com CPU")
        return "cpu"


# ==============================================================================
# WRAPPER COM PROGRESSO
# ==============================================================================

class XGBComProgresso(XGBClassifier, ClassifierMixin):
    """
    Wrapper que atualiza a barra de progresso a cada fit() chamado pelo GridSearchCV.
    """
    _estimator_type = "classifier"
    pbar: tqdm = None

    def fit(self, X, y, **kwargs):
        result = super().fit(X, y, **kwargs)
        if XGBComProgresso.pbar is not None:
            XGBComProgresso.pbar.update(1)
        return result


# ==============================================================================
# TREINAMENTO DE UM PIPELINE
# ==============================================================================

def treinar(branch: str, selecao: str, idx: int, total: int, device: str) -> dict:
    nome = f"{branch}_{selecao}"

    print("\n" + "=" * 70)
    print(f"🚀 [{idx}/{total}] Pipeline: {branch.upper()} + {SELECAO_LABEL[selecao]}")
    print(f"   Dispositivo : {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
    print(f"   Combinações : {TOTAL_COMBINACOES} × 5 folds = {TOTAL_FITS} treinamentos")
    print("=" * 70)

    dataset_path = os.path.join(FEATURES_DIR, f"dataset_{nome}.csv")
    if not os.path.exists(dataset_path):
        print(f"  ❌ Dataset não encontrado: {dataset_path}")
        return {"pipeline": nome, "status": "dataset não encontrado"}

    df = pd.read_csv(dataset_path)
    
    # <-- FILTRO DEFINITIVO: Remove todas as estatísticas descritivas e áreas para evitar Data Leakage
    colunas_vazadas = [
        c for c in df.columns 
        if c.startswith("desc_") or c.endswith("_area_norm") or c in ("sample_id", "label")
    ]
    
    feature_cols = [c for c in df.columns if c not in colunas_vazadas]
    
    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    
    # Extração da coluna de identificação para agrupamento
    grupos = df["sample_id"].values 

    print(f"  📥 {X.shape[0]} amostras × {X.shape[1]} features puras (Filtro Anti-Leakage Aplicado)")
    print(f"     Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------------------------------------------------------------------
    # GRID SEARCH COM BARRA DE PROGRESSO
    # ------------------------------------------------------------------
    cv     = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) 
    n_jobs = 1 if device == "cuda" else -1

    XGBComProgresso.pbar = tqdm(
        total=TOTAL_FITS,
        desc=f"  Grid Search [{nome}]",
        unit="fit",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    grid = GridSearchCV(
        estimator=XGBComProgresso(
            objective="binary:logistic",
            eval_metric="logloss",
            device=device,
            random_state=42,
        ),
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
    )

    t_inicio = time.time()

    with parallel_backend("threading" if device == "cuda" else "loky"):
        # Passando os grupos mapeados para o método fit
        grid.fit(X, y, groups=grupos) 

    XGBComProgresso.pbar.close()
    XGBComProgresso.pbar = None

    t_decorrido = time.time() - t_inicio
    best_model  = grid.best_estimator_

    print(f"  ✅ Concluído em {formatar_tempo(t_decorrido)}")
    print(f"  ✅ Melhores hiperparâmetros: {grid.best_params_}")

    # ------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------

    # Métricas do Cross-Validation (avaliação confiável — out-of-fold)
    cv_f1_mean = round(grid.cv_results_["mean_test_score"][grid.best_index_], 4)
    cv_f1_std  = round(grid.cv_results_["std_test_score"][grid.best_index_],  4)

    # Métricas in-sample (treino completo — apenas para diagnóstico de overfitting)
    # Muda temporariamente para CPU para evitar o warning de DMatrix
    best_model.set_params(device="cpu")

    y_pred  = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]

    # Retorna o modelo para o dispositivo original
    best_model.set_params(device=device)

    metricas = {
        "pipeline":     nome,
        "branch":       branch,
        "selecao":      SELECAO_LABEL[selecao],
        "device":       device,
        "cv_f1_mean":   cv_f1_mean,
        "cv_f1_std":    cv_f1_std,
        "acuracia":     round(accuracy_score(y, y_pred), 4),
        "f1":           round(f1_score(y, y_pred, zero_division=0), 4),
        "precisao":     round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":       round(recall_score(y, y_pred, zero_division=0), 4),
        "roc_auc":      round(roc_auc_score(y, y_proba), 4),
        "n_features":   X.shape[1],
        "tempo_s":      round(t_decorrido, 1),
        **{f"param_{k}": v for k, v in grid.best_params_.items()},
    }

    print(f"\n  📊 Métricas [{nome.upper()}]")
    print(f"     ── Cross-Validation (out-of-fold, confiável) ──")
    print(f"     cv_f1_mean : {cv_f1_mean} (±{cv_f1_std})")
    print(f"     ── In-sample / treino completo (diagnóstico) ──")
    for k in ("acuracia", "f1", "precisao", "recall", "roc_auc"):
        print(f"     {k:<10}: {metricas[k]}")
    print(f"     Matriz de confusão:\n{confusion_matrix(y, y_pred)}")

    # ------------------------------------------------------------------
    # SALVAR & DETECTAR VAZAMENTO
    # ------------------------------------------------------------------
    best_model.get_booster().save_model(
        os.path.join(MODEL_DIR, f"xgb_{nome}.json")
    )
    
    pd.DataFrame(grid.cv_results_).to_csv(
        os.path.join(RESULTS_DIR, f"grid_results_{nome}.csv"), index=False
    )

    df_importances = pd.DataFrame({
        "feature":      feature_cols,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    df_importances.to_csv(
        os.path.join(RESULTS_DIR, f"feature_importance_{nome}.csv"), index=False
    )

    # Alerta de Vazamento de Dados
    top_feature = df_importances.iloc[0]
    if top_feature["importance"] > 0.80:
        print(f"\n  🚨 ALERTA DE VAZAMENTO (DATA LEAKAGE) 🚨")
        print(f"     A feature '{top_feature['feature']}' tem {top_feature['importance']*100:.1f}% de importância.")
        print(f"     Ela provavelmente é um 'gabarito' disfarçado. Verifique a base!\n")

    print(f"  💾 Modelo salvo: xgb_{nome}.json")
    return metricas


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    total    = len(PIPELINES)
    t_global = time.time()
    tempos   = []

    print("🧬 Treinamento — 6 pipelines independentes")
    print("   MSC   × MI  |  MSC   × XGBoost Importance")
    print("   Log   × MI  |  Log   × XGBoost Importance")
    print("   Deriv × MI  |  Deriv × XGBoost Importance")
    print(f"\n   Combinações por pipeline : {TOTAL_COMBINACOES} × 5 folds = {TOTAL_FITS}")
    print(f"   Total de treinamentos    : {TOTAL_FITS * total}\n")

    device = detectar_dispositivo()

    # Barra de progresso geral (pipelines)
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
        print(f"\n  ⏱️  Tempo deste pipeline : {formatar_tempo(elapsed)}")
        print(f"  ⏱️  Tempo total até agora: {formatar_tempo(time.time() - t_global)}")
        if idx < total:
            print(f"  ⏱️  Estimativa restante  : {formatar_tempo(restante)} ({total - idx} pipelines)")

    pbar_global.close()

    # ------------------------------------------------------------------
    # RELATÓRIO FINAL
    # ------------------------------------------------------------------
    t_total = time.time() - t_global

    print("\n" + "=" * 70)
    print("📊 COMPARAÇÃO FINAL — 6 pipelines")
    print("=" * 70)

    cols_resumo = ["pipeline", "branch", "selecao", "device",
                   "cv_f1_mean", "cv_f1_std",
                   "acuracia", "f1", "precisao", "recall", "roc_auc",
                   "n_features", "tempo_s"]
    resultados_validos = [r for r in resultados if "acuracia" in r]
    if not resultados_validos:
        print("\n⚠️  Nenhum pipeline concluído com sucesso.")
        return
    df_result = (
        pd.DataFrame(resultados_validos)[cols_resumo]
        .sort_values("cv_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    print(df_result.to_string(index=False))

    melhor = df_result.iloc[0]
    print(f"\n🏆 Melhor pipeline  : {melhor['pipeline'].upper()}")
    print(f"   Transformação    : {melhor['branch'].upper()}")
    print(f"   Seleção          : {melhor['selecao']}")
    print(f"   CV F1 (out-fold) : {melhor['cv_f1_mean']} (±{melhor['cv_f1_std']})")
    print(f"   Acurácia (treino): {melhor['acuracia']}")
    print(f"   F1   (treino)    : {melhor['f1']}")
    print(f"   ROC-AUC (treino) : {melhor['roc_auc']}")
    print(f"\n⏱️  Tempo total      : {formatar_tempo(t_total)}")

    relatorio = os.path.join(RESULTS_DIR, "comparacao_pipelines.csv")
    df_result.to_csv(relatorio, index=False)
    print(f"✅ Relatório salvo em: {relatorio}")


if __name__ == "__main__":
    main()