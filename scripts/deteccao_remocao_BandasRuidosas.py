import pandas as pd
import os

input_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_derivada_primeira_ordem.csv"
output_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_sem_bandas_ruidosas.csv"

def remover_bandas_ruidosas(input_path: str, output_path: str, limiar_var=1e-6):
    df = pd.read_csv(input_path)

    # Calcula a variância de cada banda
    variancias = df.var()

    # Filtra bandas com variância maior que o limiar
    bandas_boas = variancias[variancias > limiar_var].index
    df_filtrado = df[bandas_boas]

    # Salva
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtrado.to_csv(output_path, index=False)

    print("✅ Bandas ruidosas removidas. Arquivo salvo em:")
    print(output_path)
    
remover_bandas_ruidosas(input_csv, output_csv)
