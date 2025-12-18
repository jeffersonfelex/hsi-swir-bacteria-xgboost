import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

# 📂 Diretórios
BASE_DIR = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ml"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_xgb_model.json")

def main():

    print("📥 Carregando train.csv...")
    df = pd.read_csv(TRAIN_PATH)

    X_train = df.drop(columns=["label"])
    y_train = df["label"]

    # Modelo base
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )

    # Espaço de busca
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.3]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        verbose=1
    )

    print("🚀 Iniciando Grid Search...")
    grid.fit(X_train, y_train)

    print(f"\n✅ Melhor combinação encontrada: {grid.best_params_}")

    best_model = grid.best_estimator_
    best_model.save_model(MODEL_PATH)

    print(f"💾 Modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()
