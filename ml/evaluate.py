import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from xgboost import XGBClassifier
from utils import ensure_dir, save_json, load_model_xgb, plot_feature_importance

# 📂 Diretórios
BASE_DIR   = r"C:\Users\Jefferson\Documents\GitHub\ic-2025\ml"
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
ensure_dir(REPORT_DIR)

TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_xgb_model.json")

def main():

    print("📥 Carregando test.csv...")
    df = pd.read_csv(TEST_PATH)

    X_test = df.drop(columns=["label"])
    y_test = df["label"]

    print("📥 Carregando modelo salvo...")
    model = load_model_xgb(MODEL_PATH)

    # ============================
    # Predição
    # ============================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ============================
    # Métricas
    # ============================
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    print("\n📊 MÉTRICAS:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    save_json(metrics, os.path.join(REPORT_DIR, "metrics.json"))
    print("💾 metrics.json salvo.")

    # ============================
    # Curva ROC
    # ============================
    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Curva ROC - XGBoost")
    plt.grid(True, alpha=0.3)

    roc_path = os.path.join(REPORT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 ROC curve salva em: {roc_path}")

    # ============================
    # FEATURE IMPORTANCE
    # ============================
    importance_df = plot_feature_importance(
        model,
        X_test.columns,
        output_path=os.path.join(REPORT_DIR, "feature_importance.png")
    )

    importance_df.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)
    print("📁 Feature importance salva.")

if __name__ == "__main__":
    main()
