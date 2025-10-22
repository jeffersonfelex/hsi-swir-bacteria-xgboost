import os
import pandas as pd
import numpy as np

KEEP_INTERMEDIATE = False  # ❌ Remove ROI após gerar MSC

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_roi.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_msc_signal.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df.values
    ref = np.mean(data, axis=0)
    corrected = [(s - np.polyfit(ref, s, 1)[1]) / np.polyfit(ref, s, 1)[0] for s in data]

    pd.DataFrame(corrected, columns=df.columns).to_csv(output_path, index=False)
    print(f"✅ MSC Signal aplicado: {output_path}")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")
