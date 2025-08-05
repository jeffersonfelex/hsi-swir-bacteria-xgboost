import pandas as pd
import numpy as np
import os

def remove_noisy_bands(input_path: str, output_path: str, limiar: float = 1e-4, faixa_min: float = 750, faixa_max: float = 2400):
    df = pd.read_csv(input_path)

    try:
        colunas_float = df.columns.astype(float)
    except ValueError:
        raise ValueError("As colunas do DataFrame devem representar comprimentos de onda e ser convertíveis para float.")

    variancia = df.var(axis=0)
    filtro_combinado = (variancia > limiar) & (colunas_float > faixa_min) & (colunas_float < faixa_max)
    bandas_selecionadas = df.loc[:, filtro_combinado]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bandas_selecionadas.to_csv(output_path, index=False)
    print(f"Bandas ruidosas removidas e CSV salvo em: {output_path}")
