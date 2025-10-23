import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

XGB_TOP_N = 50

def selecionar_por_xgboost(X: np.ndarray, y: np.ndarray, feature_names):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    importance = model.feature_importances_
    idx_sorted = np.argsort(importance)[::-1][:XGB_TOP_N]
    return feature_names[idx_sorted], importance[idx_sorted]

def run(input_csv_path: str, mi_features_path: str, output_dir: str):
    print(f"📥 [XGBoost] Lendo dados de {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # ✅ Corrigido para coluna certa
    mi_bands = pd.read_csv(mi_features_path)["band_name"].tolist()

    # Filtrar somente colunas numéricas que estão em mi_bands
    valid_bands = [b for b in mi_bands if b in df.columns]
    X = df[valid_bands].select_dtypes(include=[np.number]).values
    y = df["label"].values

    xgb_features, xgb_scores = selecionar_por_xgboost(X, y, np.array(valid_bands))

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "bands_xgb.csv")
    pd.DataFrame({"band_name": xgb_features, "importance": xgb_scores}).to_csv(path, index=False)

    print(f"✅ [XGBoost] Seleção final de {len(xgb_features)} bandas salva em {path}")
    return xgb_features
