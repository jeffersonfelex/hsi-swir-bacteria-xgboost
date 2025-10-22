import os
import pandas as pd
from chemotools.derivative import SavitzkyGolay

KEEP_INTERMEDIATE = False  # ❌ Remove Log após derivada

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_log.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    deriv = SavitzkyGolay(window_size=15, polynomial_order=2)
    transformed = deriv.fit_transform(df.values)

    pd.DataFrame(transformed, columns=df.columns).to_csv(output_path, index=False)
    print(f"✅ Derivada aplicada: {output_path}")

    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")
