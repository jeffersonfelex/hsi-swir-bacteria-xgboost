import os
import pandas as pd
from etl import (
    pre_processing_image,
    msc_signal,
    log_transform,
    derivative_transform,
    descriptive_stats,
    spectral_features,
    index_features,
    feature_selection_mi,
    feature_selection_xgboost,
    label_by_atcc
)

BASE_DIR     = r"C:\Users\Jefferson\Documents\GitHub\ic-2025"
HSI_ROOT     = os.path.join(BASE_DIR, "dataSet", "hsi_original")
OUTPUT_BASE  = os.path.join(BASE_DIR, "dataSet", "data-preProcessed", "pre_processing")
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


def roi_valido(csv_path: str) -> bool:
    """Verifica se o CSV do ROI existe e tem dados reais."""
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path, nrows=1)
        return not df.empty and df.shape[1] > 0
    except Exception:
        return False


def extrair_features(espectros, transformacao_fn):
    """Aplica transformação + engenharia de atributos."""
    esp = transformacao_fn(espectros)
    return pd.concat([
        descriptive_stats(esp),
        spectral_features(esp),
        index_features(esp),
    ], axis=1)


def aplicar_selecao(df_consolidado, nome_dataset):
    """
    Aplica MI e XGBoost importance de forma independente sobre o mesmo dataset,
    gerando dois CSVs separados.
    """
    feature_cols = [c for c in df_consolidado.columns if c not in ("sample_id", "label")]
    X = df_consolidado[feature_cols].fillna(0).values
    y = df_consolidado["label"].values

    print(f"\n  📊 Seleção de features: {nome_dataset.upper()}")
    print(f"     {X.shape[0]} amostras × {X.shape[1]} features")

    # PIPELINE A — Mutual Information
    X_mi, features_mi = feature_selection_mi(X, y, feature_cols)
    df_mi = pd.DataFrame(X_mi, columns=features_mi)
    df_mi.insert(0, "sample_id", df_consolidado["sample_id"].values)
    df_mi["label"] = y
    saida_mi = os.path.join(FEATURES_DIR, f"dataset_{nome_dataset}_mi.csv")
    df_mi.to_csv(saida_mi, index=False)
    print(f"  ✅ MI  → {saida_mi} ({len(features_mi)} features)")

    # PIPELINE B — XGBoost Feature Importance
    X_xgb, features_xgb = feature_selection_xgboost(X, y, feature_cols)
    df_xgb = pd.DataFrame(X_xgb, columns=features_xgb)
    df_xgb.insert(0, "sample_id", df_consolidado["sample_id"].values)
    df_xgb["label"] = y
    saida_xgb = os.path.join(FEATURES_DIR, f"dataset_{nome_dataset}_xgb.csv")
    df_xgb.to_csv(saida_xgb, index=False)
    print(f"  ✅ XGB → {saida_xgb} ({len(features_xgb)} features)")


def main():
    pastas = sorted([
        nome for nome in os.listdir(HSI_ROOT)
        if os.path.isdir(os.path.join(HSI_ROOT, nome))
    ])
    print(f"📂 {len(pastas)} imagens encontradas")

    frames_msc   = []
    frames_log   = []
    frames_deriv = []

    for folder_name in pastas:
        print(f"\n🔬 Processando: {folder_name}")

        # ------------------------------------------------------------------
        # ETAPA 0 — Filtro ROI
        # ------------------------------------------------------------------
        csv_roi = os.path.join(HSI_ROOT, f"{folder_name}_roi_filtrado.csv")

        if not roi_valido(csv_roi):
            pre_processing_image(output_dir=HSI_ROOT, folder_name=folder_name)

        if not roi_valido(csv_roi):
            print(f"  ⚠️  ROI inválido ou não detectado. Pulando {folder_name}.")
            continue

        espectros = pd.read_csv(csv_roi).values
        print(f"  ✅ ROI carregado: {espectros.shape}")

        # ------------------------------------------------------------------
        # BRANCH MSC → engenharia de atributos
        # ------------------------------------------------------------------
        df_msc = extrair_features(espectros, msc_signal)
        df_msc.insert(0, "sample_id", folder_name)
        frames_msc.append(df_msc)

        # ------------------------------------------------------------------
        # BRANCH Log(1/R) → engenharia de atributos
        # ------------------------------------------------------------------
        df_log = extrair_features(espectros, log_transform)
        df_log.insert(0, "sample_id", folder_name)
        frames_log.append(df_log)

        # ------------------------------------------------------------------
        # BRANCH Savitzky-Golay → engenharia de atributos
        # ------------------------------------------------------------------
        df_deriv = extrair_features(espectros, derivative_transform)
        df_deriv.insert(0, "sample_id", folder_name)
        frames_deriv.append(df_deriv)

    # ----------------------------------------------------------------------
    # CONSOLIDAÇÃO + ROTULAGEM + SELEÇÃO
    # ----------------------------------------------------------------------
    branches = [
        ("msc",   frames_msc),
        ("log",   frames_log),
        ("deriv", frames_deriv),
    ]

    for nome, frames in branches:
        if not frames:
            print(f"\n⚠️  Nenhuma amostra para o branch {nome.upper()}.")
            continue

        df = pd.concat(frames, ignore_index=True)
        df = label_by_atcc(df)

        print(f"\n{'='*60}")
        print(f"Branch {nome.upper()} — {df.shape[0]} amostras × {df.shape[1]} colunas")
        print(f"Classes: {df['label'].value_counts().to_dict()}")

        aplicar_selecao(df, nome)

    print(f"\n✅ Pipeline concluído — 6 datasets gerados em: {FEATURES_DIR}")
    print("   msc_mi.csv   | msc_xgb.csv")
    print("   log_mi.csv   | log_xgb.csv")
    print("   deriv_mi.csv | deriv_xgb.csv")


if __name__ == "__main__":
    main()