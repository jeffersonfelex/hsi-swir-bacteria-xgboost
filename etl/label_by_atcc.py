import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

AMOSTRAS_ATCC = ["ATCC13", "ATCC16", "ATCC27"]

def gerar_labels_por_atcc(df: pd.DataFrame, limiar_sim=0.9) -> pd.DataFrame:
    """
    Gera labels binárias com base na similaridade com cada amostra ATCC individualmente.
    Se a similaridade com QUALQUER ATCC for >= limiar, a amostra recebe label 1.
    """

    # 🔸 Seleciona apenas colunas numéricas (features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "label"]
    df[feature_cols] = df[feature_cols].fillna(0)

    if "filename" not in df.columns:
        raise ValueError("Coluna 'filename' não encontrada no dataset.")

    # 🔸 Filtra amostras ATCC
    atcc_data = df[df["filename"].apply(lambda x: any(a.lower() in x.lower() for a in AMOSTRAS_ATCC))]
    if atcc_data.empty:
        raise ValueError("Nenhuma amostra ATCC encontrada para gerar labels.")

    # 🔸 Calcula similaridade com cada ATCC individualmente
    all_similarities = []
    X = df[feature_cols].values

    for _, row in atcc_data.iterrows():
        ref_vector = row[feature_cols].values.reshape(1, -1)
        sim = cosine_similarity(X, ref_vector).flatten()
        all_similarities.append(sim)

    # 🔸 Usa o valor máximo da similaridade entre todas as ATCCs
    sim_max = np.max(np.vstack(all_similarities), axis=0)

    # 🔸 Define label = 1 se a similaridade com qualquer ATCC for >= limiar
    df["label"] = (sim_max >= limiar_sim).astype(int)

    print(f"📊 Similaridade coseno - min: {sim_max.min():.4f}, max: {sim_max.max():.4f}, média: {sim_max.mean():.4f}")
    print(f"✅ Labels gerados com limiar {limiar_sim}: {df['label'].sum()} positivos de {len(df)}")

    return df
