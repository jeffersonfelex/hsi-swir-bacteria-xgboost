import os
import glob
import shutil
import pandas as pd
from etl import (
    pre_processing_image, 
    msc_signal,
    log_transform,
    derivative_transform,
    descriptive_stats,
    spectral_features,
    index_features,
    combine_features,
    feature_selection_mi,
    feature_selection_xgboost,
    label_by_atcc
)

BASE_DIR = "C:/Users/Jefferson/Documents/GitHub/ic-2025"
HSI_ROOT = f"{BASE_DIR}/dataSet/hsi_original"
OUTPUT_BASE = f"{BASE_DIR}/dataSet/data-preProcessed/pre_processing"
FEATURES_DIR = f"{BASE_DIR}/features_selected"


def combine_all_feature_csvs(features_dir: str, combined_filename: str = "dataset_final.csv"):
    print("Combinando arquivos de features em dataset único...")
    
    feature_files = glob.glob(os.path.join(features_dir, "*_features.csv"))
    
    if not feature_files:
        feature_files = glob.glob(os.path.join(OUTPUT_BASE, "*_features.csv"))
    
    if not feature_files:
        raise FileNotFoundError(
            f"Nenhum arquivo *_features.csv encontrado em:\n"
            f"  - {features_dir}\n"
            f"  - {OUTPUT_BASE}"
        )

    print(f"   Encontrados {len(feature_files)} arquivos de features")
    
    dfs = []
    for f in feature_files:
        df = pd.read_csv(f)
        sample_name = os.path.basename(f).replace("_features.csv", "")
        df["filename"] = sample_name
        dfs.append(df)
        print(f"   ✅ {sample_name}: {len(df)} amostras, {len(df.columns)} features")

    combined_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(features_dir, combined_filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"✅ Dataset combinado salvo em: {output_path}")
    print(f"   Total: {len(combined_df)} amostras, {len(combined_df.columns)} colunas")
    
    return output_path


def csv_has_data(filepath: str) -> bool:
    """Verifica se um arquivo CSV existe e tem dados válidos."""
    if not os.path.exists(filepath):
        return False
    
    if os.path.getsize(filepath) == 0:
        return False
    
    try:
        df = pd.read_csv(filepath)
        return not df.empty and len(df.columns) > 0
    except Exception:
        return False


def ensure_derivada_exists(output_dir: str, folder_name: str, backup_path: str):
    """
    Garante que o arquivo derivada existe, restaurando do backup se necessário.
    """
    deriv_path = os.path.join(output_dir, f"{folder_name}_derivada.csv")
    
    if not os.path.exists(deriv_path) and os.path.exists(backup_path):
        shutil.copy2(backup_path, deriv_path)
        print(f"      🔄 Restaurado: {os.path.basename(deriv_path)}")
        return True
    return os.path.exists(deriv_path)


def main():
    print("🚀 Iniciando pipeline completa...\n")
    print(f"📁 Diretório base: {BASE_DIR}")
    print(f"📁 HSI root: {HSI_ROOT}")
    print(f"📁 Output: {OUTPUT_BASE}")
    print(f"📁 Features: {FEATURES_DIR}\n")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    if not os.path.exists(HSI_ROOT):
        raise FileNotFoundError(f"Diretório HSI não encontrado: {HSI_ROOT}")
    
    folders = [f for f in os.listdir(HSI_ROOT) if os.path.isdir(os.path.join(HSI_ROOT, f))]
    
    if not folders:
        raise FileNotFoundError(f"Nenhuma pasta de amostra encontrada em: {HSI_ROOT}")
    
    print(f"📂 Encontradas {len(folders)} amostras para processar\n")
    
    processed_count = 0
    error_count = 0

    for folder in folders:
        try:
            print(f"\n{'='*70}")
            print(f"📂 PROCESSANDO: {folder}")
            print(f"{'='*70}\n")
            
            # Verificar se arquivo HDR existe
            hdr_path = os.path.join(HSI_ROOT, folder, "capture", f"{folder}.hdr")
            if not os.path.exists(hdr_path):
                hdr_path_alt = os.path.join(HSI_ROOT, folder, f"{folder}.hdr")
                if os.path.exists(hdr_path_alt):
                    hdr_path = hdr_path_alt
                else:
                    print(f"⚠️ Arquivo HDR não encontrado para {folder}")
                    print(f"   Pulando...\n")
                    error_count += 1
                    continue
            
            print(f"   HDR encontrado: {hdr_path}\n")

            # =====================================================
            # 📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas
            # =====================================================
            print("📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas")
            pre_processing_image.run(HSI_ROOT, folder)
            
            src_roi = os.path.join(HSI_ROOT, f"{folder}_roi_filtrado.csv")
            dst_roi = os.path.join(OUTPUT_BASE, f"{folder}_roi.csv")
            
            if os.path.exists(src_roi):
                shutil.move(src_roi, dst_roi)
                print(f"   📝 Movido ROI para: {dst_roi}")
            else:
                print(f"   ⚠️ Arquivo ROI não gerado, pulando amostra...")
                error_count += 1
                continue
            
            if not csv_has_data(dst_roi):
                print(f"   ⚠️ ROI sem dados válidos para {folder}, pulando amostra...")
                if os.path.exists(dst_roi):
                    os.remove(dst_roi)
                error_count += 1
                continue

            # =====================================================
            # 📍 ETAPA 2: Multiplicative Scatter Correction (MSC)
            # =====================================================
            print("\n📍 ETAPA 2: Multiplicative Scatter Correction (MSC)")
            msc_signal.run(OUTPUT_BASE, folder)
            
            msc_file = os.path.join(OUTPUT_BASE, f"{folder}_msc_signal.csv")
            if not csv_has_data(msc_file):
                print(f"   ⚠️ MSC não gerou dados válidos, pulando amostra...")
                error_count += 1
                continue

            # =====================================================
            # 📍 ETAPA 3: Transformações espectrais
            # =====================================================
            print("\n📍 ETAPA 3: Transformações espectrais")
            
            print("   3.1 - Transformada Log(1/R)")
            log_transform.run(OUTPUT_BASE, folder)
            
            log_file = os.path.join(OUTPUT_BASE, f"{folder}_log.csv")
            if not csv_has_data(log_file):
                print(f"   ⚠️ Log transform não gerou dados válidos, pulando amostra...")
                error_count += 1
                continue
            
            print("   3.2 - Derivada Savitzky-Golay (1ª ordem)")
            derivative_transform.run(OUTPUT_BASE, folder)
            
            deriv_file = os.path.join(OUTPUT_BASE, f"{folder}_derivada.csv")
            if not csv_has_data(deriv_file):
                print(f"   ⚠️ Derivada não gerou dados válidos, pulando amostra...")
                error_count += 1
                continue

            # =====================================================
            # 📍 ETAPA 4: Extração de features
            # =====================================================
            print("\n📍 ETAPA 4: Extração de features")
            
            # ✅ CORREÇÃO: Criar backup PERMANENTE do arquivo derivada
            backup_deriv = os.path.join(OUTPUT_BASE, f"{folder}_derivada_BACKUP.csv")
            shutil.copy2(deriv_file, backup_deriv)
            print(f"   💾 Backup criado: {os.path.basename(backup_deriv)}")
            
            # 4.1 - Features descritivas
            print("   4.1 - Estatísticas descritivas")
            ensure_derivada_exists(OUTPUT_BASE, folder, backup_deriv)
            descriptive_stats.run(OUTPUT_BASE, folder)
            
            # 4.2 - Features espectrais
            print("   4.2 - Features espectrais")
            ensure_derivada_exists(OUTPUT_BASE, folder, backup_deriv)
            spectral_features.run(OUTPUT_BASE, folder)
            
            # 4.3 - Índices espectrais
            print("   4.3 - Índices espectrais SWIR")
            ensure_derivada_exists(OUTPUT_BASE, folder, backup_deriv)
            index_features.run(OUTPUT_BASE, folder)
            
            # Limpar backup
            if os.path.exists(backup_deriv):
                os.remove(backup_deriv)
                print(f"   🧹 Backup removido")
            
            # 4.4 - Combinar todas as features
            print("   4.4 - Combinando features")
            combine_features.run(OUTPUT_BASE, folder, remove_inputs=True, move_outputs=False)

            print(f"\n✅ Finalizado com sucesso: {folder}")
            processed_count += 1

        except Exception as e:
            print(f"\n❌ Erro ao processar {folder}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # ======================
    # Resumo do processamento
    # ======================
    print(f"\n{'='*70}")
    print(f"📊 RESUMO DO PRÉ-PROCESSAMENTO")
    print(f"{'='*70}")
    print(f"   Total de amostras: {len(folders)}")
    print(f"   Processadas com sucesso: {processed_count}")
    print(f"   Erros: {error_count}")
    
    if processed_count == 0:
        print("\n❌ Nenhuma amostra foi processada com sucesso!")
        return

    # ======================
    # 2️⃣ Combinação de todos os features
    # ======================
    print(f"\n{'='*70}")
    print("📦 ETAPA 5: Combinando features de todas as amostras")
    print(f"{'='*70}\n")
    
    try:
        final_dataset_path = combine_all_feature_csvs(OUTPUT_BASE)
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
        return

    # ======================
    # 3️⃣ Geração de labels ATCC
    # ======================
    print(f"\n{'='*70}")
    print("🏷️ ETAPA 6: Gerando labels via similaridade ATCC")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(final_dataset_path)
    
    try:
        df = label_by_atcc.gerar_labels_por_atcc(df, limiar_sim=0.30)
    except Exception as e:
        print(f"⚠️ Erro ao gerar labels ATCC: {e}")
        print("   Criando labels fictícios (0) para todas as amostras...")
        df["label"] = 0

    # Limpeza de colunas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if 'label' not in numeric_cols:
        numeric_cols.append('label')
    
    cols_to_keep = ["filename"] + [c for c in numeric_cols if c != "filename"]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    df = df[cols_to_keep]
    df = df.fillna(0)
    df.to_csv(final_dataset_path, index=False)
    
    print(f"✅ Labels gerados e dataset limpo salvo em: {final_dataset_path}")
    print(f"   Amostras: {len(df)}")
    print(f"   Features: {len(df.columns) - 2}")
    print(f"   Labels positivos: {df['label'].sum()}")
    print(f"   Labels negativos: {len(df) - df['label'].sum()}")

    # ======================
    # 4️⃣ Seleção de bandas
    # ======================
    print(f"\n{'='*70}")
    print("📊 ETAPA 7: Seleção de features")
    print(f"{'='*70}\n")
    
    if df['label'].nunique() < 2:
        print("⚠️ Labels sem variabilidade! Pulando seleção de features...")
        print("   (Todas as amostras têm o mesmo label)")
    else:
        print("7.1 - Seleção via Mutual Information...")
        feature_selection_mi.run(final_dataset_path, FEATURES_DIR)

        print("\n7.2 - Seleção via XGBoost feature importance...")
        mi_features_path = os.path.join(FEATURES_DIR, "features_selected_mi.csv")
        
        if os.path.exists(mi_features_path):
            feature_selection_xgboost.run(final_dataset_path, mi_features_path, FEATURES_DIR)
        else:
            print(f"   ⚠️ Arquivo MI não encontrado: {mi_features_path}")

    # ======================
    # ✅ Finalização
    # ======================
    print(f"\n{'='*70}")
    print("🏁 PIPELINE COMPLETA FINALIZADA COM SUCESSO!")
    print(f"{'='*70}")
    print(f"\n📁 Arquivos gerados:")
    print(f"   - Dataset final: {final_dataset_path}")
    print(f"   - Features MI: {FEATURES_DIR}/features_selected_mi.csv")
    print(f"   - Features XGBoost: {FEATURES_DIR}/features_selected_xgboost.csv")
    print(f"\n🚀 Próximos passos:")
    print(f"   1. Execute ml/train.py para treinar o modelo")
    print(f"   2. Execute ml/evaluate.py para avaliar métricas")


if __name__ == "__main__":
    main()