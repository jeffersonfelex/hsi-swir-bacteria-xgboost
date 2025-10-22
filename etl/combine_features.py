import os
import pandas as pd

KEEP_INTERMEDIATE = False  # ❌ Remove sub-features após combinar

def run(output_dir: str, folder_name: str):
    desc = os.path.join(output_dir, f"{folder_name}_features_desc.csv")
    spec = os.path.join(output_dir, f"{folder_name}_features_spectral.csv")
    idx = os.path.join(output_dir, f"{folder_name}_features_index.csv")
    out = os.path.join(output_dir, f"{folder_name}_features.csv")

    dfs = []
    for p in [desc, spec, idx]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if not df.empty:
                dfs.append(df)

    if not dfs:
        pd.DataFrame().to_csv(out, index=False)
        print(f"⚠️ Nenhuma feature encontrada. Criado arquivo vazio {out}")
        return

    combined = pd.concat(dfs, axis=1)
    combined.to_csv(out, index=False)
    print(f"✅ Features combinadas: {out}")

    if not KEEP_INTERMEDIATE:
        for p in [desc, spec, idx]:
            if os.path.exists(p):
                os.remove(p)
                print(f"🧹 Arquivo removido: {p}")
