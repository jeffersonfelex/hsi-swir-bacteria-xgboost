import os
import pandas as pd

def run(output_dir: str, folder_name: str):
    # Antes estava assim:
    # input_path = os.path.join(output_dir, f"{folder_name}_roi.csv")
    # Agora corrigido:
    input_path = os.path.join(output_dir, f"{folder_name}_filtrado.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Pulando features descritivas.")
        return

    df = pd.read_csv(input_path)
    stats = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
        "median": df.median()
    }).T  # transpor se necessário

    output_path = os.path.join(output_dir, f"{folder_name}_features_descriptive.csv")
    stats.to_csv(output_path)
    print(f"✅ Features descritivas salvas: {output_path}")
