import os
import pandas as pd
import numpy as np
from chemotools.derivative import SavitzkyGolay

KEEP_INTERMEDIATE = False  # ❌ Remove Log após derivada

def run(output_dir: str, folder_name: str):
    """
    Aplica derivada de primeira ordem usando filtro Savitzky-Golay.
    
    Destacar inflexões espectrais e minimizar efeito de baseline.
    Parâmetros: janela=15, polinômio grau 2, 1ª derivada.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome base do arquivo
    """
    input_path = os.path.join(output_dir, f"{folder_name}_log.csv")
    output_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")

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
    n_samples, n_bands = data.shape

    # Validar dimensões mínimas para Savitzky-Golay
    window_size = 15
    polynomial_order = 2
    
    if n_bands < window_size:
        print(f"⚠️ Savitzky-Golay requer pelo menos {window_size} bandas.")
        print(f"   Encontrado: {n_bands} bandas. Ajustando window_size...")
        window_size = n_bands if n_bands % 2 == 1 else n_bands - 1
        
        if window_size < 5:
            print(f"❌ Número de bandas insuficiente para derivada ({n_bands} < 5)")
            pd.DataFrame().to_csv(output_path, index=False)
            return

    # Garantir que window_size seja ímpar
    if window_size % 2 == 0:
        window_size += 1

    # Aplicar derivada Savitzky-Golay (1ª ordem)
    try:
        deriv = SavitzkyGolay(
            window_size=window_size,
            polynomial_order=polynomial_order,
            derivative_order=1  # ✅ 1ª derivada (explícito)
        )
        transformed = deriv.fit_transform(data)
    except Exception as e:
        print(f"❌ Erro ao aplicar derivada Savitzky-Golay: {e}")
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Salvar resultado
    df_transformed = pd.DataFrame(transformed, columns=numeric_cols)
    df_transformed.to_csv(output_path, index=False)
    
    print(f"✅ Derivada Savitzky-Golay aplicada: {output_path}")
    print(f"   Dimensões: {transformed.shape[0]} amostras × {transformed.shape[1]} bandas")
    print(f"   Parâmetros: window={window_size}, poly_order={polynomial_order}, deriv_order=1")
    print(f"   Range: [{transformed.min():.6f}, {transformed.max():.6f}]")

    # Limpar arquivo intermediário
    if not KEEP_INTERMEDIATE and os.path.exists(input_path):
        os.remove(input_path)
        print(f"🧹 Arquivo removido: {input_path}")