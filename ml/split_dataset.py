import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 📁 Caminhos
BASE_DIR = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025"
DATASET_PATH = f"{BASE_DIR}/dataSet/data-preProcessed/processed/features/dataset_final.csv"
OUTPUT_DIR = f"{BASE_DIR}/ml/data"

def main():
    print(f"📥 Lendo dataset de: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # 📊 Verifica se 'label' existe
    if "label" not in df.columns:
        raise ValueError("❌ Coluna 'label' não encontrada no dataset_final.csv.")

    # 🧹 Separa features e target
    X = df.drop(columns=["label", "filename"], errors="ignore")
    y = df["label"]

    # ✂️ Divide em treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,          # 📌 30% teste
        random_state=42,
        stratify=y
    )

    # 📂 Cria diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 💾 Salva conjuntos
    train_df = X_train.copy()
    train_df["label"] = y_train
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)

    test_df = X_test.copy()
    test_df["label"] = y_test
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"✅ Treino salvo em: {os.path.join(OUTPUT_DIR, 'train.csv')} ({len(train_df)} linhas)")
    print(f"✅ Teste salvo em: {os.path.join(OUTPUT_DIR, 'test.csv')} ({len(test_df)} linhas)")

    # 📊 Estatísticas de distribuição
    print("\n📊 Distribuição de classes:")
    print("Treino:\n", y_train.value_counts())
    print("Teste:\n", y_test.value_counts())

if __name__ == "__main__":
    main()
