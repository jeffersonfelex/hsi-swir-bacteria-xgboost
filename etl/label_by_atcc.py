import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Identificadores de amostras ATCC conhecidas
AMOSTRAS_ATCC = ["ATCC13", "ATCC16", "ATCC27"]

def gerar_labels_por_atcc(df: pd.DataFrame, limiar_sim=0.85, plot_distribution=True) -> pd.DataFrame:
    """
    Gera labels binárias com base na similaridade espectral com amostras ATCC.
    
    ATENÇÃO: Este método assume que amostras ATCC representam cepas resistentes.
    Validar com dados microbiológicos (antibiograma/qPCR).
    
    Args:
        df: DataFrame com features e coluna 'filename'
        limiar_sim: Limiar de similaridade coseno (0-1)
        plot_distribution: Se True, plota distribuição de similaridades
    
    Returns:
        DataFrame com coluna 'label' adicionada
    """
    
    print(f"\n{'='*70}")
    print("🏷️  GERAÇÃO DE LABELS POR SIMILARIDADE ATCC")
    print(f"{'='*70}")
    
    # Verificar coluna filename
    if "filename" not in df.columns:
        raise ValueError("Coluna 'filename' não encontrada no dataset.")
    
    # Selecionar colunas numéricas (features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["label"]]
    
    if len(feature_cols) == 0:
        raise ValueError("Nenhuma feature numérica encontrada!")
    
    # Tratar valores ausentes
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Filtrar amostras ATCC (referências)
    atcc_mask = df["filename"].str.contains('|'.join(AMOSTRAS_ATCC), case=False, na=False)
    atcc_data = df[atcc_mask]
    
    if atcc_data.empty:
        raise ValueError(f"Nenhuma amostra ATCC encontrada! Esperado: {AMOSTRAS_ATCC}")
    
    print(f"✅ Amostras ATCC encontradas: {len(atcc_data)}")
    print(f"   IDs: {atcc_data['filename'].tolist()}")
    
    # Calcular similaridade com cada ATCC
    X = df[feature_cols].values
    all_similarities = []
    
    for idx, row in atcc_data.iterrows():
        ref_vector = row[feature_cols].values.reshape(1, -1)
        sim = cosine_similarity(X, ref_vector).flatten()
        all_similarities.append(sim)
    
    # Usar máximo de similaridade com qualquer ATCC
    sim_matrix = np.vstack(all_similarities)
    sim_max = np.max(sim_matrix, axis=0)
    
    # Estatísticas de similaridade
    print(f"\n📊 Distribuição de Similaridade:")
    print(f"   Mínimo:   {sim_max.min():.4f}")
    print(f"   Máximo:   {sim_max.max():.4f}")
    print(f"   Média:    {sim_max.mean():.4f}")
    print(f"   Mediana:  {np.median(sim_max):.4f}")
    print(f"   Q1:       {np.percentile(sim_max, 25):.4f}")
    print(f"   Q3:       {np.percentile(sim_max, 75):.4f}")
    
    # Gerar labels
    df["label"] = (sim_max >= limiar_sim).astype(int)
    df["similarity_atcc"] = sim_max  # Manter para análise
    
    n_positivos = df["label"].sum()
    n_total = len(df)
    
    print(f"\n✅ Labels gerados com limiar {limiar_sim}:")
    print(f"   Positivos (resistentes): {n_positivos} ({n_positivos/n_total*100:.1f}%)")
    print(f"   Negativos (sensíveis):   {n_total - n_positivos} ({(n_total - n_positivos)/n_total*100:.1f}%)")
    
    # Plot distribuição (opcional)
    if plot_distribution:
        plt.figure(figsize=(10, 6))
        plt.hist(sim_max, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(limiar_sim, color='red', linestyle='--', linewidth=2, label=f'Limiar = {limiar_sim}')
        plt.xlabel('Similaridade Coseno com ATCC')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Similaridade Espectral com Amostras ATCC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('distribuicao_similaridade_atcc.png', dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo: distribuicao_similaridade_atcc.png")
        plt.close()
    
    print(f"{'='*70}\n")
    
    return df