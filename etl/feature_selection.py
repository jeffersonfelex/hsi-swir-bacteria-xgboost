import os
import pandas as pd

KEEP_INTERMEDIATE = False  # ❌ Remove features após seleção final

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_features.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_selected_features.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Pulando seleção de features.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Pulando seleção de features.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Por enquanto apenas copia
    df.to_csv(output_path, index=False)
    print(f"✅ Features copiadas para {output_path}")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")
