import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List


def combine_features_for_sample(
    output_dir: str, 
    folder_name: str, 
    remove_inputs: bool = True, 
    move_outputs: bool = False
) -> Optional[str]:
    """
    Combina os arquivos de features gerados para uma amostra específica:
      - {folder_name}_features_descriptive.csv
      - {folder_name}_features_spectral.csv
      - {folder_name}_features_index.csv
    
    Salva o resultado em:
      - {output_dir}/{folder_name}_features.csv

    Args:
        output_dir: Diretório onde os arquivos estão localizados.
        folder_name: Nome base da amostra.
        remove_inputs: Se True, remove os arquivos intermediários de features individuais.
        move_outputs: Flag reservada para compatibilidade de pipeline.

    Returns:
        Caminho do arquivo {folder_name}_features.csv gerado, ou None em caso de falha.
    """
    desc_path = os.path.join(output_dir, f"{folder_name}_features_descriptive.csv")
    spec_path = os.path.join(output_dir, f"{folder_name}_features_spectral.csv")
    idx_path  = os.path.join(output_dir, f"{folder_name}_features_index.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_features.csv")

    inputs = [p for p in [desc_path, spec_path, idx_path] if os.path.exists(p)]
    if not inputs:
        print(f"⚠️  Nenhum arquivo de feature encontrado para {folder_name} em {output_dir}")
        return None

    dfs: List[pd.DataFrame] = []
    for path in [desc_path, spec_path, idx_path]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"⚠️  Erro ao ler {path}: {e}")

    if not dfs:
        print(f"❌ Nenhum dado válido encontrado para {folder_name}")
        return None

    # Concatenação horizontal (lado a lado) das features
    combined_df = pd.concat(dfs, axis=1)

    # Remover colunas duplicadas se houver
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    combined_df.to_csv(output_path, index=False)
    print(f"✅ Features combinadas para {folder_name}: {output_path} ({combined_df.shape[1]} features, {combined_df.shape[0]} amostras)")

    # Limpeza de arquivos intermediários
    if remove_inputs:
        for path in [desc_path, spec_path, idx_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"⚠️  Não foi possível remover intermediário {path}: {e}")

    return output_path


# Alias padrão de chamada dos módulos ETL
run = combine_features_for_sample
