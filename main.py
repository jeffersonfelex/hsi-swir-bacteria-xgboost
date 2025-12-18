import os
import glob
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

BASE_DIR = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025"
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
        # Extrair nome da amostra do arquivo
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


# ============================================
# 🔧 Função auxiliar para renomear arquivo
# ============================================
def rename_roi_file(output_dir: str, folder_name: str):
    """
    CORREÇÃO: Renomeia {folder}_roi_filtrado.csv para {folder}_roi.csv
    para compatibilidade com msc_signal.run()
    """
    src = os.path.join(output_dir, f"{folder_name}_roi_filtrado.csv")
    dst = os.path.join(output_dir, f"{folder_name}_roi.csv")
    
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
        print(f"   📝 Renomeado: {os.path.basename(src)} → {os.path.basename(dst)}")
        return True
    return False


# ============================================
# 🔧 Função para copiar arquivo derivada
# ============================================
def preserve_derivada_file(output_dir: str, folder_name: str):
    """
    CORREÇÃO: Cria cópia do arquivo derivada antes da extração de features
    para garantir que todos os módulos tenham acesso ao arquivo.
    """
    import shutil
    
    src = os.path.join(output_dir, f"{folder_name}_derivada.csv")
    
    if os.path.exists(src):
        # Criar backup se necessário
        backup = os.path.join(output_dir, f"{folder_name}_derivada_backup.csv")
        shutil.copy2(src, backup)
        return True
    return False


def restore_derivada_file(output_dir: str, folder_name: str):
    """Restaura arquivo derivada do backup se necessário"""
    src = os.path.join(output_dir, f"{folder_name}_derivada.csv")
    backup = os.path.join(output_dir, f"{folder_name}_derivada_backup.csv")
    
    if not os.path.exists(src) and os.path.exists(backup):
        os.rename(backup, src)
        return True
    elif os.path.exists(backup):
        os.remove(backup)
    return False


# ============================================
# 🧠 Pipeline Principal (CORRIGIDA)
# ============================================
def main():
    print("🚀 Iniciando pipeline completa...\n")
    print(f"📁 Diretório base: {BASE_DIR}")
    print(f"📁 HSI root: {HSI_ROOT}")
    print(f"📁 Output: {OUTPUT_BASE}")
    print(f"📁 Features: {FEATURES_DIR}\n")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Listar pastas de amostras HSI
    if not os.path.exists(HSI_ROOT):
        raise FileNotFoundError(f"Diretório HSI não encontrado: {HSI_ROOT}")
    
    folders = [f for f in os.listdir(HSI_ROOT) if os.path.isdir(os.path.join(HSI_ROOT, f))]
    
    if not folders:
        raise FileNotFoundError(f"Nenhuma pasta de amostra encontrada em: {HSI_ROOT}")
    
    print(f"📂 Encontradas {len(folders)} amostras para processar\n")
    
    processed_count = 0
    error_count = 0

    # ======================
    # 1️⃣ Pré-processamento e extração de features (por amostra)
    # ======================
    for folder in folders:
        try:
            print(f"\n{'='*70}")
            print(f"📂 PROCESSANDO: {folder}")
            print(f"{'='*70}\n")
            
            # Verificar se arquivo HDR existe
            hdr_path = os.path.join(HSI_ROOT, folder, "capture", f"{folder}.hdr")
            if not os.path.exists(hdr_path):
                # Tentar caminho alternativo
                hdr_path_alt = os.path.join(HSI_ROOT, folder, f"{folder}.hdr")
                if os.path.exists(hdr_path_alt):
                    hdr_path = hdr_path_alt
                else:
                    print(f"⚠️ Arquivo HDR não encontrado para {folder}")
                    print(f"   Tentativas:")
                    print(f"   - {hdr_path}")
                    print(f"   - {hdr_path_alt}")
                    print(f"   Pulando...\n")
                    error_count += 1
                    continue
            
            print(f"   HDR encontrado: {hdr_path}\n")

            # =====================================================
            # 📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas
            # =====================================================
            print("📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas")
            
            # NOTA: pre_processing_image.run() precisa do caminho correto
            # O módulo atual usa output_dir como base para construir o caminho do HDR
            # Então passamos HSI_ROOT como output_dir aqui
            pre_processing_image.run(HSI_ROOT, folder)
            
            # CORREÇÃO: Renomear arquivo para compatibilidade
            # pre_processing_image gera: {folder}_roi_filtrado.csv
            # msc_signal espera: {folder}_roi.csv
            src_roi = os.path.join(HSI_ROOT, f"{folder}_roi_filtrado.csv")
            dst_roi = os.path.join(OUTPUT_BASE, f"{folder}_roi.csv")
            
            if os.path.exists(src_roi):
                import shutil
                shutil.move(src_roi, dst_roi)
                print(f"   📝 Movido ROI para: {dst_roi}")
            else:
                print(f"   ⚠️ Arquivo ROI não gerado, pulando amostra...")
                error_count += 1
                continue

            # =====================================================
            # 📍 ETAPA 2: Multiplicative Scatter Correction (MSC)
            # =====================================================
            print("\n📍 ETAPA 2: Multiplicative Scatter Correction (MSC)")
            msc_signal.run(OUTPUT_BASE, folder)
            
            # Verificar se MSC foi aplicado
            msc_file = os.path.join(OUTPUT_BASE, f"{folder}_msc_signal.csv")
            if not os.path.exists(msc_file) or os.path.getsize(msc_file) == 0:
                print(f"   ⚠️ MSC não gerou dados válidos, pulando amostra...")
                error_count += 1
                continue

            # =====================================================
            # 📍 ETAPA 3: Transformações espectrais
            # =====================================================
            print("\n📍 ETAPA 3: Transformações espectrais")
            
            # 3.1 - Log transform
            print("   3.1 - Transformada Log(1/R)")
            log_transform.run(OUTPUT_BASE, folder)
            
            # 3.2 - Derivada Savitzky-Golay
            print("   3.2 - Derivada Savitzky-Golay (1ª ordem)")
            derivative_transform.run(OUTPUT_BASE, folder)
            
            # Verificar se derivada foi gerada
            deriv_file = os.path.join(OUTPUT_BASE, f"{folder}_derivada.csv")
            if not os.path.exists(deriv_file) or os.path.getsize(deriv_file) == 0:
                print(f"   ⚠️ Derivada não gerou dados válidos, pulando amostra...")
                error_count += 1
                continue

            # =====================================================
            # 📍 ETAPA 4: Extração de features
            # =====================================================
            print("\n📍 ETAPA 4: Extração de features")
            
            # CORREÇÃO: Preservar arquivo derivada para todos os módulos
            # Alguns módulos removem o arquivo após processamento
            preserve_derivada_file(OUTPUT_BASE, folder)
            
            # 4.1 - Features descritivas (média, std, curtose, skewness)
            print("   4.1 - Estatísticas descritivas")
            descriptive_stats.run(OUTPUT_BASE, folder)
            restore_derivada_file(OUTPUT_BASE, folder)
            
            # 4.2 - Features espectrais (gradientes, razões, área)
            print("   4.2 - Features espectrais")
            spectral_features.run(OUTPUT_BASE, folder)
            restore_derivada_file(OUTPUT_BASE, folder)
            
            # 4.3 - Índices espectrais (NDI, SR, profundidades)
            print("   4.3 - Índices espectrais SWIR")
            index_features.run(OUTPUT_BASE, folder)
            
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
        df = label_by_atcc.gerar_labels_por_atcc(df, limiar_sim=0.9)
    except Exception as e:
        print(f"⚠️ Erro ao gerar labels ATCC: {e}")
        print("   Criando labels fictícios (0) para todas as amostras...")
        df["label"] = 0

    # ======================
    # 🧼 Limpeza de colunas não numéricas
    # ======================
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Garantir que 'label' está nas colunas
    if 'label' not in numeric_cols:
        numeric_cols.append('label')
    
    # Manter filename para rastreabilidade
    cols_to_keep = ["filename"] + [c for c in numeric_cols if c != "filename"]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    df = df[cols_to_keep]
    df = df.fillna(0)
    df.to_csv(final_dataset_path, index=False)
    
    print(f"✅ Labels gerados e dataset limpo salvo em: {final_dataset_path}")
    print(f"   Amostras: {len(df)}")
    print(f"   Features: {len(df.columns) - 2}")  # -2 para filename e label
    print(f"   Labels positivos: {df['label'].sum()}")
    print(f"   Labels negativos: {len(df) - df['label'].sum()}")

    # ======================
    # 4️⃣ Seleção de bandas (MI + XGBoost)
    # ======================
    print(f"\n{'='*70}")
    print("📊 ETAPA 7: Seleção de features")
    print(f"{'='*70}\n")
    
    # Verificar se há variabilidade nas labels
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