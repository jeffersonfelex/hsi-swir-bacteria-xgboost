import os
import shutil
import pandas as pd

def _rm(path: str):
    """Remove arquivo se existir"""
    if os.path.exists(path):
        os.remove(path)
        print(f"🧹 Arquivo removido: {os.path.basename(path)}")

def _ensure_dir(path: str):
    """Cria diretório se não existir"""
    os.makedirs(path, exist_ok=True)

def _move_file(src: str, dst_dir: str):
    """Move arquivo, sobrescrevendo se já existir"""
    if not os.path.exists(src):
        return
    
    _ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    
    # Remove destino se já existir
    if os.path.exists(dst):
        os.remove(dst)
    
    shutil.move(src, dst)
    print(f"📦 Movido: {os.path.basename(src)} → {dst_dir}")

def run(output_dir: str, folder_name: str, remove_inputs: bool = True, move_outputs: bool = False):
    """
    Combina features descritivas, espectrais e de índices em um único CSV.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome da amostra
        remove_inputs: Se True, remove CSVs individuais de features
        move_outputs: Se True, move arquivos para estrutura organizada
    """
    
    # Caminhos dos arquivos de features
    desc_path = os.path.join(output_dir, f"{folder_name}_features_descriptive.csv")
    spec_path = os.path.join(output_dir, f"{folder_name}_features_spectral.csv")
    idx_path  = os.path.join(output_dir, f"{folder_name}_features_index.csv")
    out_path  = os.path.join(output_dir, f"{folder_name}_features.csv")

    # Carregar features existentes
    inputs = []
    input_paths = []
    
    for p in [desc_path, spec_path, idx_path]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if not df.empty:
                inputs.append(df)
                input_paths.append(p)
        else:
            print(f"⚠️ {os.path.basename(p)} não encontrado. Pulando...")

    # Verificar se há features para combinar
    if not inputs:
        print(f"⚠️ Nenhuma feature encontrada para {folder_name}. Criando saída vazia.")
        pd.DataFrame().to_csv(out_path, index=False)
        return

    # Verificar consistência no número de linhas
    n_rows = [len(df) for df in inputs]
    if len(set(n_rows)) > 1:
        print(f"❌ ERRO: DataFrames com números diferentes de linhas: {n_rows}")
        print(f"   Arquivos: {[os.path.basename(p) for p in input_paths]}")
        pd.DataFrame().to_csv(out_path, index=False)
        return

    # Combinar features (juntar colunas)
    combined = pd.concat(inputs, axis=1)
    
    # Adicionar coluna filename se não existir
    if "filename" not in combined.columns:
        combined.insert(0, "filename", folder_name)
    
    # Remover colunas duplicadas (manter primeira ocorrência)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # Salvar features combinadas
    combined.to_csv(out_path, index=False)
    
    print(f"✅ Features combinadas: {out_path}")
    print(f"   Dimensões: {combined.shape[0]} amostras × {combined.shape[1]} features")

    # Limpar arquivos intermediários de features
    if remove_inputs:
        for p in input_paths:
            _rm(p)

    # Organizar arquivos em estrutura de diretórios
    if move_outputs:
        base_processed = os.path.join(os.path.dirname(output_dir), "processed")
        features_dir = os.path.join(base_processed, "features")
        
        _ensure_dir(features_dir)
        
        # Mover arquivo de features combinadas
        _move_file(out_path, features_dir)
        
        print(f"📁 Arquivos organizados em {base_processed}")