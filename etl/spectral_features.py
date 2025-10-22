import os
import numpy as np
import pandas as pd

def run(output_dir: str, folder_name: str):
    input_path = os.path.join(output_dir, f"{folder_name}_filtrado.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features_spectral.csv")

    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Criando saída vazia.")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df.values

    # Derivadas locais — média do gradiente por banda
    grad = np.mean(np.gradient(data, axis=1), axis=0)

    # Razão espectral simples entre primeira e última banda
    ratios = []
    if data.shape[1] >= 2:
        b1 = data[:, 0]
        b2 = data[:, -1]
        ratio = (b2 - b1) / (b2 + b1 + 1e-8)
        ratios.append(np.mean(ratio))
    else:
        ratios.append(np.nan)

    features = np.concatenate([grad, np.array(ratios)])
    n_bandas = data.shape[1]
    colunas = [f"grad_b{i}" for i in range(n_bandas)] + ["ratio_first_last"]

    pd.DataFrame([features], columns=colunas).to_csv(output_path, index=False)
    print(f"✅ Features espectrais salvas: {output_path}")
