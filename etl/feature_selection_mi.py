import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

MI_TOP_N = 100  # ou ajuste conforme desejar

def run(input_csv_path: str, output_dir: str):
    print(f"📥 [MI] Lendo dataset de {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # ✅ Mantém apenas colunas numéricas para MI
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "label" not in numeric_cols:
        raise ValueError("Coluna 'label' não encontrada ou não numérica no dataset.")

    feature_cols = [c for c in numeric_cols if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # 📊 Calcula informação mútua
    mi_scores = mutual_info_classif(X, y, random_state=42)
    idx_sorted = np.argsort(mi_scores)[::-1][:MI_TOP_N]

    top_features = np.array(feature_cols)[idx_sorted]
    top_scores = mi_scores[idx_sorted]

    # 📁 Salva os resultados
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features_selected_mi.csv")
    pd.DataFrame({"band_name": top_features, "mi_score": top_scores}).to_csv(output_path, index=False)

    print(f"✅ [MI] {len(top_features)} bandas selecionadas salvas em {output_path}")
    return top_features
