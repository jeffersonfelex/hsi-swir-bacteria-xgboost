import numpy as np
import pandas as pd
import os

def apply_log_transform(input_path: str, output_path: str):
    
    df = pd.read_csv(input_path)
    X = df.values

    # Evita log de zero ou valores negativos
    X[X <= 0] = 1e-6
    X_log = -np.log(X)

    df_log = pd.DataFrame(X_log, columns=df.columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_log.to_csv(output_path, index=False)
    print(f"Transformação log aplicada e salva em: {output_path}")
