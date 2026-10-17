"""
Avaliacao no CONJUNTO DE TESTE EXTERNO (separado por imagem em split_dataset.py).

Para cada pipeline:
  - carrega ml/models/xgb_<nome>.json (treinado por ml/train.py);
  - reajusta StandardScaler + PCA no conjunto de TREINO correspondente;
  - avalia no conjunto de TESTE no nivel de PIXEL e no nivel de IMAGEM
    (voto majoritario dos pixels de cada imagem);
  - gera curvas ROC e Precisao-Recall.

Saidas em ml/reports/:
  metrics_<nome>.json, curves_<nome>.png, comparacao_teste.csv, barplot_comparativo_teste.png
"""

import os
import sys
import json
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay,
)
from xgboost import XGBClassifier

BASE_DIR = os.getenv(
    "IC2025_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
MODEL_DIR  = os.path.join(BASE_DIR, "ml", "models")
DATA_DIR   = os.path.join(BASE_DIR, "ml", "data")
REPORT_DIR = os.path.join(BASE_DIR, "ml", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

PIPELINES = [
    "msc_mi", "msc_xgb",
    "log_mi", "log_xgb",
    "deriv_mi", "deriv_xgb",
    "raw_mi", "raw_xgb",
]


def _safe(fn, *a, **kw):
    try:
        return float(fn(*a, **kw))
    except Exception:
        return float("nan")


def _feature_cols(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if not (c.startswith("desc_") or c.endswith("_area_norm") or c in ("sample_id", "label"))
    ]


def carregar(pipeline_name: str):
    model_path = os.path.join(MODEL_DIR, f"xgb_{pipeline_name}.json")
    test_path  = os.path.join(DATA_DIR,  f"test_{pipeline_name}.csv")
    train_path = os.path.join(DATA_DIR,  f"train_{pipeline_name}.csv")

    for p in (model_path, test_path, train_path):
        if not os.path.exists(p):
            print(f"  [pulado] arquivo ausente: {os.path.relpath(p, BASE_DIR)}")
            return None

    df_test  = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
    feats = _feature_cols(df_test)

    xgb = XGBClassifier()
    xgb.load_model(model_path)
    n_comp = xgb.get_booster().num_features()

    scaler = StandardScaler().fit(df_train[feats].fillna(0).values)
    pca = PCA(n_components=n_comp, random_state=42).fit(
        scaler.transform(df_train[feats].fillna(0).values)
    )
    pipe = Pipeline([("scaler", scaler), ("pca", pca), ("xgb", xgb)])
    return pipe, df_test, feats


def avaliar(pipeline_name: str) -> dict:
    print(f"\n{'='*60}\nAvaliando: {pipeline_name}")
    loaded = carregar(pipeline_name)
    if loaded is None:
        return {}
    pipe, df_test, feats = loaded

    X = df_test[feats].fillna(0).values
    y = df_test["label"].values
    grupos = df_test["sample_id"].values

    Xp = pipe.named_steps["pca"].transform(pipe.named_steps["scaler"].transform(X))
    y_pred  = pipe.named_steps["xgb"].predict(Xp)
    y_proba = pipe.named_steps["xgb"].predict_proba(Xp)[:, 1]

    uma_classe = len(np.unique(y)) < 2
    m = {
        "pipeline": pipeline_name,
        "n_pixels_teste": int(len(y)),
        "n_imagens_teste": int(len(np.unique(grupos))),
        "n_imagens_resistentes_teste": int(pd.Series(y, index=grupos).groupby(level=0).first().sum()),
        "recall":    round(_safe(recall_score, y, y_pred, zero_division=0), 4),
        "precision": round(_safe(precision_score, y, y_pred, zero_division=0), 4),
        "f1":        round(_safe(f1_score, y, y_pred, zero_division=0), 4),
        "auprc":     round(_safe(average_precision_score, y, y_proba), 4),
        "roc_auc":   round(_safe(roc_auc_score, y, y_proba), 4),
        "accuracy":  round(_safe(accuracy_score, y, y_pred), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        "aviso": "conjunto de teste com uma unica classe" if uma_classe else "",
    }
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    m["n_falsos_negativos"] = fn
    m["n_falsos_positivos"] = fp

    # ---- nivel de imagem (voto majoritario) ----
    dfp = pd.DataFrame({"sample_id": grupos, "y_true": y, "y_pred": y_pred, "y_proba": y_proba})
    img = dfp.groupby("sample_id").agg(
        label_real=("y_true", "first"),
        pred_majoritario=("y_pred", lambda x: int(x.mode()[0])),
        proba_media=("y_proba", "mean"),
    ).reset_index()
    m["recall_por_imagem"]    = round(_safe(recall_score, img["label_real"], img["pred_majoritario"], zero_division=0), 4)
    m["precision_por_imagem"] = round(_safe(precision_score, img["label_real"], img["pred_majoritario"], zero_division=0), 4)
    m["f1_por_imagem"]        = round(_safe(f1_score, img["label_real"], img["pred_majoritario"], zero_division=0), 4)
    m["auprc_por_imagem"]     = round(_safe(average_precision_score, img["label_real"], img["proba_media"]), 4)
    m["confusion_matrix_imagem"] = confusion_matrix(img["label_real"], img["pred_majoritario"]).tolist()

    print(f"  pixels {m['n_pixels_teste']} | imagens {m['n_imagens_teste']} "
          f"(resistentes: {m['n_imagens_resistentes_teste']})")
    print(f"  PIXEL  -> recall {m['recall']} | precision {m['precision']} | f1 {m['f1']} | "
          f"auprc {m['auprc']} | roc_auc {m['roc_auc']}")
    print(f"  IMAGEM -> recall {m['recall_por_imagem']} | precision {m['precision_por_imagem']} | "
          f"f1 {m['f1_por_imagem']}")
    if m["aviso"]:
        print(f"  AVISO: {m['aviso']} — metricas dependentes de ranking podem ser NaN.")

    with open(os.path.join(REPORT_DIR, f"metrics_{pipeline_name}.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    if not uma_classe:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Evaluation — {pipeline_name.upper()} (test set, pixel level)", fontsize=12)
        RocCurveDisplay.from_predictions(y, y_proba, ax=axes[0], name=pipeline_name)
        axes[0].set_title(f"ROC (AUC={m['roc_auc']})"); axes[0].grid(alpha=0.3)
        PrecisionRecallDisplay.from_predictions(y, y_proba, ax=axes[1], name=pipeline_name)
        axes[1].set_title(f"Precision-Recall (AUPRC={m['auprc']})"); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"curves_{pipeline_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    return m


def main():
    print("Avaliacao no conjunto de teste externo (separado por imagem)")
    print("Metrica primaria: RECALL | secundaria: AUPRC\n")

    todos = [r for r in (avaliar(p) for p in PIPELINES) if r]
    if not todos:
        print("\nNenhum pipeline avaliado (rode split_dataset.py e train.py antes).")
        return

    cols = [
        "pipeline", "n_imagens_teste", "n_imagens_resistentes_teste",
        "recall", "precision", "f1", "auprc", "roc_auc",
        "recall_por_imagem", "precision_por_imagem", "f1_por_imagem", "auprc_por_imagem",
        "n_falsos_negativos", "n_falsos_positivos",
    ]
    df = pd.DataFrame(todos)[cols].sort_values("recall", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 78)
    print("COMPARACAO FINAL — CONJUNTO DE TESTE")
    print("=" * 78)
    print(df.to_string(index=False))

    rng = df["recall"].max() - df["recall"].min()
    if len(df) > 1 and rng < 0.10:
        print("\n  NOTA: diferenca de recall entre pipelines < 0.10 — ranking NAO robusto "
              "(poucas imagens de teste).")

    out = os.path.join(REPORT_DIR, "comparacao_teste.csv")
    df.to_csv(out, index=False)
    print(f"\nTabela comparativa salva: {out}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Test set — pipeline comparison (pixel level)", fontsize=12)
    for ax, col, tit in zip(axes, ["recall", "auprc"], ["Recall (primary)", "AUPRC"]):
        ax.bar(df["pipeline"], df[col], color="steelblue", alpha=0.8, edgecolor="black")
        ax.set_title(tit); ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(df[col].values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "barplot_comparativo_teste.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Grafico comparativo salvo: barplot_comparativo_teste.png")


if __name__ == "__main__":
    main()
