import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Diretórios
BASE_DIR   = r"C:\Users\Jefferson\Documents\GitHub\ic-2025"
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")
MODEL_DIR  = os.path.join(BASE_DIR, "ml", "models")
REPORT_DIR = os.path.join(BASE_DIR, "ml", "reports")
DATA_DIR   = os.path.join(BASE_DIR, "ml", "data")

os.makedirs(REPORT_DIR, exist_ok=True)

PIPELINES = [
    "msc_mi", "msc_xgb",
    "log_mi", "log_xgb",
    "deriv_mi", "deriv_xgb",
    "raw_mi", "raw_xgb",
]


def ic95(scores: list) -> tuple:
    """Intervalo de confiança 95% via t-distribution."""
    n = len(scores)
    if n < 2:
        return (float("nan"), float("nan"))
    se  = stats.sem(scores)
    h   = se * stats.t.ppf(0.975, df=n - 1)
    m   = np.mean(scores)
    return (round(float(m - h), 4), round(float(m + h), 4))


def carregar_modelo_e_dados(pipeline_name: str):
    """
    Carrega o modelo XGBoost salvo e os dados de teste (separados por imagem).
    Retorna (pipeline_sklearn, X_test, y_test, grupos_test) ou None se arquivos faltam.
    """
    model_path = os.path.join(MODEL_DIR, f"xgb_{pipeline_name}.json")
    test_path  = os.path.join(DATA_DIR,  f"test_{pipeline_name}.csv")

    if not os.path.exists(model_path):
        print(f"  Modelo não encontrado: {model_path}")
        return None

    if not os.path.exists(test_path):
        print(f"  Arquivo de teste não encontrado: {test_path}")
        print(f"  Execute ml/split_dataset.py primeiro.")
        return None

    df_test = pd.read_csv(test_path)

    colunas_vazadas = [
        c for c in df_test.columns
        if c.startswith("desc_") or c.endswith("_area_norm") or c in ("sample_id", "label")
    ]
    feature_cols = [c for c in df_test.columns if c not in colunas_vazadas]

    X_test  = df_test[feature_cols].fillna(0).values
    y_test  = df_test["label"].values
    grupos  = df_test["sample_id"].values

    # Reconstrói pipeline (scaler → PCA → XGBoost) e carrega pesos do XGBoost
    xgb = XGBClassifier()
    xgb.load_model(model_path)

    pipeline_sklearn = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=0.95, random_state=42)),
        ("xgb",    xgb),
    ])

    # Ajusta scaler e PCA no conjunto de treino correspondente
    train_path = os.path.join(DATA_DIR, f"train_{pipeline_name}.csv")
    if not os.path.exists(train_path):
        print(f"  Arquivo de treino não encontrado para ajustar PCA: {train_path}")
        return None

    df_train = pd.read_csv(train_path)
    X_train  = df_train[feature_cols].fillna(0).values

    pipeline_sklearn.named_steps["scaler"].fit(X_train)
    X_train_scaled = pipeline_sklearn.named_steps["scaler"].transform(X_train)
    pipeline_sklearn.named_steps["pca"].fit(X_train_scaled)

    return pipeline_sklearn, X_test, y_test, grupos, feature_cols


def avaliar_pipeline(pipeline_name: str) -> dict:
    """Avalia um pipeline no conjunto de teste externo."""
    print(f"\n{'='*60}")
    print(f"Avaliando: {pipeline_name}")

    resultado = carregar_modelo_e_dados(pipeline_name)
    if resultado is None:
        return {}

    pipeline_sklearn, X_test, y_test, grupos, feature_cols = resultado

    X_scaled = pipeline_sklearn.named_steps["scaler"].transform(X_test)
    X_pca    = pipeline_sklearn.named_steps["pca"].transform(X_scaled)

    y_pred  = pipeline_sklearn.named_steps["xgb"].predict(X_pca)
    y_proba = pipeline_sklearn.named_steps["xgb"].predict_proba(X_pca)[:, 1]

    n_grupos = len(np.unique(grupos))

    # ── Métricas principais ──
    metricas = {
        "pipeline":         pipeline_name,
        "n_pixels_teste":   int(len(y_test)),
        "n_imagens_teste":  int(n_grupos),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "auprc":            round(average_precision_score(y_test, y_proba), 4),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc_auc_score(y_test, y_proba), 4),
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    print(f"  Pixels de teste  : {metricas['n_pixels_teste']}")
    print(f"  Imagens de teste : {metricas['n_imagens_teste']}")
    print(f"\n  Métrica primária (screening):")
    print(f"    Recall  : {metricas['recall']}  (prioridade — detectar resistentes)")
    print(f"    AUPRC   : {metricas['auprc']}   (área sob curva precisão-recall)")
    print(f"\n  Métricas complementares:")
    print(f"    F1      : {metricas['f1']}")
    print(f"    Precision: {metricas['precision']}")
    print(f"    ROC-AUC : {metricas['roc_auc']}")
    print(f"    Accuracy: {metricas['accuracy']}")
    print(f"\n  Matriz de confusão:\n{confusion_matrix(y_test, y_pred)}")

    # ── Análise de falhas ──
    idx_fn = np.where((y_test == 1) & (y_pred == 0))[0]  # Falsos Negativos
    idx_fp = np.where((y_test == 0) & (y_pred == 1))[0]  # Falsos Positivos
    print(f"\n  Análise de falhas:")
    print(f"    Falsos Negativos (resistentes perdidos) : {len(idx_fn)}")
    print(f"    Falsos Positivos (sensíveis classificados como resistentes): {len(idx_fp)}")

    metricas["n_falsos_negativos"] = int(len(idx_fn))
    metricas["n_falsos_positivos"] = int(len(idx_fp))

    # ── Análise por imagem (nível de grupo) ──
    # Agrega previsões por imagem (maioria de pixels)
    df_pred = pd.DataFrame({
        "sample_id": grupos,
        "y_true":    y_test,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    })
    imagem_pred = df_pred.groupby("sample_id").agg(
        label_real=("y_true", "first"),
        pred_pixel_majoritario=("y_pred", lambda x: int(x.mode()[0])),
        proba_media=("y_proba", "mean"),
    ).reset_index()

    recall_img   = recall_score(imagem_pred["label_real"], imagem_pred["pred_pixel_majoritario"], zero_division=0)
    auprc_img    = average_precision_score(imagem_pred["label_real"], imagem_pred["proba_media"])
    f1_img       = f1_score(imagem_pred["label_real"], imagem_pred["pred_pixel_majoritario"], zero_division=0)
    cm_img       = confusion_matrix(imagem_pred["label_real"], imagem_pred["pred_pixel_majoritario"])

    print(f"\n  Métricas por IMAGEM (voto majoritário de pixels):")
    print(f"    Recall  : {recall_img:.4f}")
    print(f"    AUPRC   : {auprc_img:.4f}")
    print(f"    F1      : {f1_img:.4f}")
    print(f"    Matriz  :\n{cm_img}")

    metricas["recall_por_imagem"]   = round(float(recall_img), 4)
    metricas["auprc_por_imagem"]    = round(float(auprc_img), 4)
    metricas["f1_por_imagem"]       = round(float(f1_img), 4)

    # ── Salvar métricas ──
    metricas_path = os.path.join(REPORT_DIR, f"metrics_{pipeline_name}.json")
    with open(metricas_path, "w") as f:
        json.dump(metricas, f, indent=2)
    print(f"\n  Métricas salvas: {metricas_path}")

    # ── Curva ROC ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Avaliação — {pipeline_name.upper()}", fontsize=12)

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name=pipeline_name)
    axes[0].set_title(f"Curva ROC (AUC={metricas['roc_auc']})")
    axes[0].grid(True, alpha=0.3)

    # ── Curva Precisão-Recall (AUPRC) ──
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1], name=pipeline_name)
    axes[1].set_title(f"Curva Precisão-Recall (AUPRC={metricas['auprc']})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(REPORT_DIR, f"curves_{pipeline_name}.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curvas salvas: {curves_path}")

    return metricas


def main():
    print("Avaliação nos conjuntos de teste (separados por imagem)")
    print("=" * 60)
    print("Métrica primária: RECALL (screening — detectar resistentes é prioridade)")
    print("Métrica secundária: AUPRC\n")

    todos_resultados = []

    for pipeline in PIPELINES:
        res = avaliar_pipeline(pipeline)
        if res:
            todos_resultados.append(res)

    if not todos_resultados:
        print("\nNenhum pipeline avaliado.")
        return

    # ── Tabela comparativa ──
    cols = [
        "pipeline", "n_imagens_teste",
        "recall", "auprc", "f1", "precision", "roc_auc",
        "recall_por_imagem", "auprc_por_imagem",
        "n_falsos_negativos", "n_falsos_positivos",
    ]
    df_comp = (
        pd.DataFrame(todos_resultados)[cols]
        .sort_values("recall", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL — CONJUNTO DE TESTE (por imagem)")
    print("=" * 70)
    print(df_comp.to_string(index=False))

    # Aviso sobre robustez estatística
    recalls = df_comp["recall"].values
    if len(recalls) > 1 and (recalls.max() - recalls.min()) < 0.10:
        print("\n  NOTA: Diferença entre pipelines < 0.10 em recall.")
        print("  O ranking NÃO é estatisticamente robusto com poucos grupos (imagens).")

    comp_path = os.path.join(REPORT_DIR, "comparacao_teste.csv")
    df_comp.to_csv(comp_path, index=False)
    print(f"\nTabela comparativa salva: {comp_path}")

    # ── Boxplot comparativo de recall e AUPRC ──
    if len(todos_resultados) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Comparação — Conjunto de Teste (por imagem)", fontsize=12)

        for ax, col, titulo in zip(
            axes,
            ["recall", "auprc"],
            ["Recall (Métrica Primária)", "AUPRC"]
        ):
            vals  = df_comp[col].values
            names = df_comp["pipeline"].values
            ax.bar(names, vals, color="steelblue", alpha=0.75, edgecolor="black")
            ax.set_title(titulo)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=30)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

        plt.tight_layout()
        bar_path = os.path.join(REPORT_DIR, "barplot_comparativo_teste.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gráfico comparativo salvo: {bar_path}")


if __name__ == "__main__":
    main()
