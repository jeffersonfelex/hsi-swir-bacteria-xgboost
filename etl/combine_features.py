import os
import shutil
import pandas as pd

def _rm(path: str):
    if os.path.exists(path):
        os.remove(path)
        print(f"🧹 Arquivo removido: {path}")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def run(output_dir: str, folder_name: str, remove_inputs: bool = True, move_outputs: bool = True):
    desc_path = os.path.join(output_dir, f"{folder_name}_features_descriptive.csv")
    spec_path = os.path.join(output_dir, f"{folder_name}_features_spectral.csv")
    idx_path  = os.path.join(output_dir, f"{folder_name}_features_index.csv")
    out_path  = os.path.join(output_dir, f"{folder_name}_features.csv")

    inputs = []
    for p in [desc_path, spec_path, idx_path]:
        if os.path.exists(p):
            inputs.append(pd.read_csv(p))
        else:
            print(f"⚠️ {p} não encontrado. Pulando...")

    if not inputs:
        print(f"⚠️ Nenhuma feature encontrada para {folder_name}. Pulando combinação.")
        return

    combined = pd.concat(inputs, axis=1)
    if "filename" not in combined.columns:
        combined["filename"] = folder_name
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.to_csv(out_path, index=False)
    print(f"✅ Features combinadas: {out_path}")

    # Limpa os arquivos intermediários
    if remove_inputs:
        _rm(desc_path)
        _rm(spec_path)
        _rm(idx_path)

    # Diretórios organizados
    if move_outputs:
        base_processed = os.path.join(os.path.dirname(output_dir), "processed")
        features_dir = os.path.join(base_processed, "features")
        filtrado_dir = os.path.join(base_processed, "filtrado")
        png_dir = os.path.join(base_processed, "png")

        _ensure_dir(features_dir)
        _ensure_dir(filtrado_dir)
        _ensure_dir(png_dir)

        # move features
        shutil.move(out_path, os.path.join(features_dir, os.path.basename(out_path)))

        # move filtrado se existir
        filtrado_path = os.path.join(output_dir, f"{folder_name}_filtrado.csv")
        if os.path.exists(filtrado_path):
            shutil.move(filtrado_path, os.path.join(filtrado_dir, os.path.basename(filtrado_path)))

        # move png se existir
        roi_png = os.path.join(output_dir, f"{folder_name}_roi.png")
        if os.path.exists(roi_png):
            shutil.move(roi_png, os.path.join(png_dir, os.path.basename(roi_png)))

        print(f"📁 Arquivos organizados em {base_processed}")
