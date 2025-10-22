import pandas as pd
import numpy as np
import os

KEEP_INTERMEDIATE = False  # ❌ Remove MSC após gerar Log

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_msc_signal.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_log.csv")

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
    data[data <= 0] = 1e-6
    transformed = -np.log(data)

    pd.DataFrame(transformed, columns=df.columns).to_csv(output_path, index=False)
    print(f"✅ Log transform aplicado: {output_path}")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")
