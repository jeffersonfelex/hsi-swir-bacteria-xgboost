import pandas as pd
import numpy as np
import os

KEEP_INTERMEDIATE = False  # ❌ Remove MSC após gerar Log

def apply_log_transform(data):
    """
    Aplica transformada logarítmica: Log(1/R) = -Log(R)
    
    Utilizada para realçar bandas de baixa intensidade e comprimir amplitudes,
    facilitando a modelagem de relações não lineares.
    
    Args:
        data: matriz numpy [n_samples × n_bands]
    
    Returns:
        matriz transformada numpy [n_samples × n_bands]
    """
    # Clonar dados para não modificar original
    data_clean = data.copy()
    
    # Tratar valores inválidos
    data_clean[data_clean <= 0] = 1e-6  # Evitar log(0) ou log(negativo)
    data_clean[np.isnan(data_clean)] = 1e-6  # Tratar NaN
    data_clean[np.isinf(data_clean)] = 1e-6  # Tratar infinitos
    
    # Aplicar Log(1/R) = -Log(R)
    transformed = -np.log(data_clean)
    
    return transformed


def run(output_dir: str, folder_name: str):
    """
    Aplica transformada logarítmica Log(1/R) aos dados corrigidos por MSC.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome base do arquivo
    """
    input_path = os.path.join(output_dir, f"{folder_name}_msc_signal.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_log.csv")

    # Verificar se arquivo existe
    if not os.path.exists(input_path):
        print(f"⚠️ {input_path} não encontrado. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Carregar dados
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ {input_path} vazio. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Extrair apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        print(f"⚠️ Nenhuma coluna numérica em {input_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    data = df[numeric_cols].values

    # Aplicar transformada logarítmica
    transformed = apply_log_transform(data)

    # Salvar resultado
    df_transformed = pd.DataFrame(transformed, columns=numeric_cols)
    df_transformed.to_csv(output_path, index=False)
    
    print(f"✅ Log transform aplicado: {output_path}")
    print(f"   Dimensões: {transformed.shape[0]} amostras × {transformed.shape[1]} bandas")
    print(f"   Range: [{transformed.min():.3f}, {transformed.max():.3f}]")

    # Limpar arquivo intermediário
    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")