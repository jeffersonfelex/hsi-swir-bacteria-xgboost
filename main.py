import os
from etl import (
    extract_roi,
    msc_signal,
    log_transform,
    derivative_transform,
    noise_band_removal,
    descriptive_stats,
    spectral_features,
    index_features,
    combine_features,
    feature_selection
)

# Diretórios
HSI_ROOT = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/hsi_original"
OUTPUT_BASE = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/dataSet/data-preProcessed/pre_processing"

def main():
    print("🚀 Iniciando pipeline completa...\n")

    folders = [f for f in os.listdir(HSI_ROOT) if os.path.isdir(os.path.join(HSI_ROOT, f))]

    for folder in folders:
        try:
            print(f"📂 Processando: {folder}")
            hdr_path = os.path.join(HSI_ROOT, folder, "capture", f"{folder}.hdr")
            if not os.path.exists(hdr_path):
                print(f"⚠️ Arquivo HDR não encontrado para {folder}, pulando...\n")
                continue

            # 📌 ETL
            extract_roi.run(hdr_path, OUTPUT_BASE, folder)
            msc_signal.run(OUTPUT_BASE, folder)
            log_transform.run(OUTPUT_BASE, folder)
            derivative_transform.run(OUTPUT_BASE, folder)
            noise_band_removal.run(OUTPUT_BASE, folder)

            # 🧠 Engenharia de atributos
            descriptive_stats.run(OUTPUT_BASE, folder)
            spectral_features.run(OUTPUT_BASE, folder)
            index_features.run(OUTPUT_BASE, folder)
            combine_features.run(OUTPUT_BASE, folder)

            # 🪄 Seleção de features (placeholder por enquanto)
            feature_selection.run(OUTPUT_BASE, folder)

            print(f"✅ Finalizado com sucesso: {folder}\n")

        except Exception as e:
            print(f"❌ Erro ao processar {folder}: {e}\n")

    print("🏁 Pipeline completa finalizada.")

if __name__ == "__main__":
    main()
