"""
Constrói o pipeline RAW (baseline): dados brutos do pré-processamento de imagem
direto para seleção de features e modelo, sem transformações espectrais.

Uso: python build_raw_pipeline.py
"""

import os
import glob
import pandas as pd
import numpy as np
from etl.label_by_atcc import label_by_atcc
from etl.feature_selection_mi import feature_selection_mi
from etl.feature_selection_xgboost import feature_selection_xgboost

BASE_DIR     = r"C:/Users/Jefferson/Documents/GitHub/ic-2025"
HSI_DIR      = os.path.join(BASE_DIR, "dataSet", "hsi_original")
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")

MI_TOP_N  = 100
XGB_TOP_N = 50


def build():
    print("=" * 60)
    print("Pipeline RAW (baseline) — sem transformações espectrais")
    print("=" * 60)

    roi_files = sorted(glob.glob(os.path.join(HSI_DIR, "*_roi_filtrado.csv")))

    if not roi_files:
        raise FileNotFoundError(f"Nenhum *_roi_filtrado.csv em {HSI_DIR}")

    # ── 1. Ler e filtrar CSVs válidos ──
    dfs = []
    for f in roi_files:
        if os.path.getsize(f) == 0:
            print(f"  Pulando (0 bytes): {os.path.basename(f)}")
            continue

        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            print(f"  Pulando (sem colunas): {os.path.basename(f)}")
            continue

        if df.empty or len(df.columns) < 2:
            print(f"  Pulando (vazio): {os.path.basename(f)}")
            continue

        sample_name = os.path.basename(f).replace("_roi_filtrado.csv", "")
        df["sample_id"] = sample_name
        dfs.append(df)
        print(f"  ✅ {sample_name}: {len(df)} pixels, {len(df.columns)-1} bandas")

    if not dfs:
        raise ValueError("Nenhum arquivo ROI com dados válidos.")

    # ── 2. Manter apenas bandas comuns entre todas as imagens ──
    band_sets = [set(df.columns) - {"sample_id"} for df in dfs]
    common_bands = sorted(set.intersection(*band_sets))
    print(f"\n  Bandas comuns entre {len(dfs)} imagens: {len(common_bands)}")

    cols_final = common_bands + ["sample_id"]
    combined = pd.concat([df[cols_final] for df in dfs], ignore_index=True)

    # ── 3. Labels via ATCC ──
    combined = label_by_atcc(combined)

    print(f"\n  Dataset RAW combinado: {len(combined)} pixels × {len(common_bands)} bandas")
    print(f"  Classes: {dict(zip(*np.unique(combined['label'], return_counts=True)))}")

    # ── 4. Seleção de features ──
    feature_cols = common_bands
    X = combined[feature_cols].fillna(0).values
    y = combined["label"].values

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 4.1 — Mutual Information
    print(f"\n{'='*60}")
    print("Seleção MI — Pipeline RAW")
    print("=" * 60)
    X_mi, names_mi = feature_selection_mi(X, y, feature_cols)

    df_raw_mi = combined[["sample_id"]].copy()
    for i, name in enumerate(names_mi):
        df_raw_mi[name] = X_mi[:, i]
    df_raw_mi["label"] = y

    mi_path = os.path.join(FEATURES_DIR, "dataset_raw_mi.csv")
    df_raw_mi.to_csv(mi_path, index=False)
    print(f"  ✅ Salvo: {mi_path} ({df_raw_mi.shape[1]-2} features)")

    # 4.2 — XGBoost Importance (sobre features MI)
    print(f"\n{'='*60}")
    print("Seleção XGBoost — Pipeline RAW")
    print("=" * 60)
    X_xgb, names_xgb = feature_selection_xgboost(X_mi, y, names_mi)

    df_raw_xgb = combined[["sample_id"]].copy()
    for i, name in enumerate(names_xgb):
        df_raw_xgb[name] = X_xgb[:, i]
    df_raw_xgb["label"] = y

    xgb_path = os.path.join(FEATURES_DIR, "dataset_raw_xgb.csv")
    df_raw_xgb.to_csv(xgb_path, index=False)
    print(f"  ✅ Salvo: {xgb_path} ({df_raw_xgb.shape[1]-2} features)")

    print(f"\n{'='*60}")
    print("Pipeline RAW concluído!")
    print(f"  dataset_raw_mi.csv  : {len(names_mi)} features")
    print(f"  dataset_raw_xgb.csv : {len(names_xgb)} features")
    print("=" * 60)


if __name__ == "__main__":
    build()
