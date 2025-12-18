import os
import glob
import pandas as pd
from etl import (
    pre_processing_image, 
    msc_signal,
    log_transform,
    derivative_transform,
    descriptive_stats,
    spectral_features,
    index_features,
    combine_features,
    feature_selection_mi,
    feature_selection_xgboost,
    label_by_atcc
)

# Diretórios
BASE_DIR = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025"
HSI_ROOT = f"{BASE_DIR}/dataSet/hsi_original"
OUTPUT_BASE = f"{BASE_DIR}/dataSet/data-preProcessed/pre_processing"
FEATURES_DIR = f"{BASE_DIR}/features_selected"
FINAL_DATASET_PATH = os.path.join(OUTPUT_BASE, "dataset_final.csv")

# ============================================
# 📦 Combinar todos os CSVs de features
# ============================================
def combine_all_feature_csvs(features_dir: str, combined_filename: str = "dataset_final.csv"):
    """
    Combina todos os arquivos *_features.csv da pasta processed/features
    """
    print("🧭 Combinando arquivos de features em dataset único...")
    feature_files = glob.glob(os.path.join(features_dir, "*_features.csv"))

    if not feature_files:
        raise FileNotFoundError("Nenhum arquivo *_features.csv encontrado para combinar.")

    dfs = []
    for f in feature_files:
        df = pd.read_csv(f)
        df["filename"] = os.path.basename(f).replace("_features.csv", "")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(features_dir, combined_filename)
    combined_df.to_csv(output_path, index=False)

    print(f"✅ Dataset combinado salvo em: {output_path}")
    return output_path


# ============================================
# 🧠 Pipeline Principal
# ============================================
def main():
    print("🚀 Iniciando pipeline completa...\n")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    folders = [f for f in os.listdir(HSI_ROOT) if os.path.isdir(os.path.join(HSI_ROOT, f))]

    # ======================
    # 1️⃣ Pré-processamento e extração de features
    # ======================
    for folder in folders:
        try:
            print(f"\n{'='*70}")
            print(f"📂 PROCESSANDO: {folder}")
            print(f"{'='*70}\n")
            
            hdr_path = os.path.join(HSI_ROOT, folder, "capture", f"{folder}.hdr")
            if not os.path.exists(hdr_path):
                print(f"⚠️ Arquivo HDR não encontrado para {folder}, pulando...\n")
                continue

            # 📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas
            print("📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas")
            pre_processing_image.run(OUTPUT_BASE, folder)  # Gera {folder}_roi.csv
            
            # 📍 ETAPA 2: Multiplicative Scatter Correction (MSC)
            print("\n📍 ETAPA 2: Multiplicative Scatter Correction (MSC)")
            msc_signal.run(OUTPUT_BASE, folder)  # Gera {folder}_msc_signal.csv e remove _roi.csv
            
            # 📍 ETAPA 3: Transformações adicionais
            print("\n📍 ETAPA 3: Transformações adicionais")
            log_transform.run(OUTPUT_BASE, folder)
            derivative_transform.run(OUTPUT_BASE, folder)

            # 📍 ETAPA 4: Extração de features
            print("\n📍 ETAPA 4: Extração de features")
            descriptive_stats.run(OUTPUT_BASE, folder)
            spectral_features.run(OUTPUT_BASE, folder)
            index_features.run(OUTPUT_BASE, folder)
            combine_features.run(OUTPUT_BASE, folder)

            print(f"\n✅ Finalizado com sucesso: {folder}\n")

        except Exception as e:
            print(f"❌ Erro ao processar {folder}: {e}\n")

    # ======================
    # 2️⃣ Combinação de todos os features
    # ======================
    final_dataset_path = combine_all_feature_csvs(
        os.path.join(os.path.dirname(OUTPUT_BASE), "processed/features")
    )

    # ======================
    # 3️⃣ Geração de labels ATCC
    # ======================
    df = pd.read_csv(final_dataset_path)
    df = label_by_atcc.gerar_labels_por_atcc(df, limiar_sim=0.9)

    # ======================
    # 🧼 Limpeza de colunas não numéricas
    # ======================
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Mantém filename e label
    df = df[["filename"] + numeric_cols]
    df = df.fillna(0)
    df.to_csv(final_dataset_path, index=False)
    print(f"🏷️ Labels gerados e dataset limpo salvo em: {final_dataset_path}\n")

    # ======================
    # 4️⃣ Seleção de bandas (MI + XGBoost)
    # ======================
    print("📊 Selecionando bandas via Mutual Information...")
    feature_selection_mi.run(final_dataset_path, FEATURES_DIR)

    print("🌲 Selecionando bandas via XGBoost...")
    mi_features_path = os.path.join(FEATURES_DIR, "features_selected_mi.csv")
    feature_selection_xgboost.run(final_dataset_path, mi_features_path, FEATURES_DIR)

    print("\n🏁 Pipeline completa finalizada com sucesso.")


if __name__ == "__main__":
    main()