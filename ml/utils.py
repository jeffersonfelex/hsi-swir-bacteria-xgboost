import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def load_model_xgb(path):
    model = XGBClassifier()
    model.load_model(path)
    return model

def plot_feature_importance(model, feature_names, output_path):
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")

    df = pd.DataFrame(
        importance.items(),
        columns=["feature", "gain"]
    ).sort_values(by="gain", ascending=False)

    plt.figure(figsize=(8, 12))
    plt.barh(df["feature"].head(40), df["gain"].head(40))
    plt.gca().invert_yaxis()
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Gain")
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return df
