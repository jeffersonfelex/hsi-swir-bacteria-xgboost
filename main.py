"""
Pipeline de Processamento de Imagens Hiperespectrais SWIR
=========================================================
Classificação espectral de bactérias multirresistentes com Gradient Boosting

Autor: Jefferson Faria
Projeto: Iniciação Científica 2025
"""

import os
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes
import sys
import glob
import shutil
import logging
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
import pandas as pd

# Imports dos módulos ETL
from etl import (
    pre_processing_image,
    msc_signal,
    log_transform,
    derivative_transform,
    descriptive_stats,
    spectral_features,
    index_features,
    feature_selection_mi,
    feature_selection_xgboost,
    label_by_atcc
)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
BASE_DIR     = r"C:\Users\Jefferson\Documents\GitHub\ic-2025"
HSI_ROOT     = os.path.join(BASE_DIR, "dataSet", "hsi_original")
OUTPUT_BASE  = os.path.join(BASE_DIR, "dataSet", "data-preProcessed", "pre_processing")
FEATURES_DIR = os.path.join(BASE_DIR, "features_selected")

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


def roi_valido(csv_path: str) -> bool:
    """Verifica se o CSV do ROI existe e tem dados reais."""
    if not os.path.exists(csv_path):
=======

# ============================================================================
# 📁 CONFIGURAÇÃO CENTRALIZADA
# ============================================================================

@dataclass
class Config:
    """Configurações centralizadas do pipeline."""
    
    # Diretórios principais
    BASE_DIR: str = "C:/Users/Jefferson/Documents/GitHub/ic-2025"
    
    @property
    def HSI_ROOT(self) -> str:
        return os.path.join(self.BASE_DIR, "dataSet", "hsi_original")
    
    @property
    def OUTPUT_BASE(self) -> str:
        return os.path.join(self.BASE_DIR, "dataSet", "data-preProcessed", "pre_processing")
    
    @property
    def FEATURES_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "features_selected")
    
    @property
    def LOG_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "logs")
    
    # Parâmetros do pipeline
    MI_TOP_N: int = 100
    XGB_TOP_N: int = 50
    LIMIAR_SIMILARIDADE: float = 0.30
    
    # Controle de arquivos intermediários
    KEEP_INTERMEDIATE: bool = False


=======

# ============================================================================
# 📁 CONFIGURAÇÃO CENTRALIZADA
# ============================================================================

@dataclass
class Config:
    """Configurações centralizadas do pipeline."""
    
    # Diretórios principais
    BASE_DIR: str = "C:/Users/Jefferson/Documents/GitHub/ic-2025"
    
    @property
    def HSI_ROOT(self) -> str:
        return os.path.join(self.BASE_DIR, "dataSet", "hsi_original")
    
    @property
    def OUTPUT_BASE(self) -> str:
        return os.path.join(self.BASE_DIR, "dataSet", "data-preProcessed", "pre_processing")
    
    @property
    def FEATURES_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "features_selected")
    
    @property
    def LOG_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "logs")
    
    # Parâmetros do pipeline
    MI_TOP_N: int = 100
    XGB_TOP_N: int = 50
    LIMIAR_SIMILARIDADE: float = 0.30
    
    # Controle de arquivos intermediários
    KEEP_INTERMEDIATE: bool = False


>>>>>>> Stashed changes
# Instância global de configuração
config = Config()


# ============================================================================
# 📝 CONFIGURAÇÃO DE LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configura sistema de logging com arquivo e console."""
<<<<<<< Updated upstream
    
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f"pipeline_{timestamp}.log")
    
    # Configurar logger
    logger = logging.getLogger("HSI_Pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicação de handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"📝 Log salvo em: {log_file}")
    
    return logger


# Logger global
logger = setup_logging()


# ============================================================================
# 🛠️ FUNÇÕES UTILITÁRIAS
# ============================================================================

def csv_has_data(filepath: str) -> bool:
    """
    Verifica se um arquivo CSV existe e contém dados válidos.
    
    Args:
        filepath: Caminho do arquivo CSV
        
    Returns:
        True se arquivo existe e tem dados, False caso contrário
    """
    if not os.path.exists(filepath):
>>>>>>> Stashed changes
        return False
    try:
<<<<<<< Updated upstream
        df = pd.read_csv(csv_path, nrows=1)
        return not df.empty and df.shape[1] > 0
    except Exception:
        return False


def extrair_features(espectros, transformacao_fn):
    """Aplica transformação + engenharia de atributos."""
    esp = transformacao_fn(espectros)
    return pd.concat([
        descriptive_stats(esp),
        spectral_features(esp),
        index_features(esp),
    ], axis=1)


def aplicar_selecao(df_consolidado, nome_dataset):
    """
    Aplica MI e XGBoost importance de forma independente sobre o mesmo dataset,
    gerando dois CSVs separados.
    """
    feature_cols = [c for c in df_consolidado.columns if c not in ("sample_id", "label")]
    X = df_consolidado[feature_cols].fillna(0).values
    y = df_consolidado["label"].values

    print(f"\n  📊 Seleção de features: {nome_dataset.upper()}")
    print(f"     {X.shape[0]} amostras × {X.shape[1]} features")

    # PIPELINE A — Mutual Information
    X_mi, features_mi = feature_selection_mi(X, y, feature_cols)
    df_mi = pd.DataFrame(X_mi, columns=features_mi)
    df_mi.insert(0, "sample_id", df_consolidado["sample_id"].values)
    df_mi["label"] = y
    saida_mi = os.path.join(FEATURES_DIR, f"dataset_{nome_dataset}_mi.csv")
    df_mi.to_csv(saida_mi, index=False)
    print(f"  ✅ MI  → {saida_mi} ({len(features_mi)} features)")

    # PIPELINE B — XGBoost Feature Importance
    X_xgb, features_xgb = feature_selection_xgboost(X, y, feature_cols)
    df_xgb = pd.DataFrame(X_xgb, columns=features_xgb)
    df_xgb.insert(0, "sample_id", df_consolidado["sample_id"].values)
    df_xgb["label"] = y
    saida_xgb = os.path.join(FEATURES_DIR, f"dataset_{nome_dataset}_xgb.csv")
    df_xgb.to_csv(saida_xgb, index=False)
    print(f"  ✅ XGB → {saida_xgb} ({len(features_xgb)} features)")
=======
        df = pd.read_csv(filepath)
        return not df.empty and len(df.columns) > 0
    except Exception as e:
        logger.warning(f"Erro ao ler {filepath}: {e}")
        return False


def ensure_file_exists(
    output_dir: str, 
    folder_name: str, 
    suffix: str, 
    backup_path: Optional[str] = None
) -> bool:
    """
    Garante que um arquivo existe, restaurando do backup se necessário.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome da amostra
        suffix: Sufixo do arquivo (ex: "_derivada.csv")
        backup_path: Caminho do backup (opcional)
        
    Returns:
        True se arquivo existe ou foi restaurado, False caso contrário
    """
    file_path = os.path.join(output_dir, f"{folder_name}{suffix}")
    
    if os.path.exists(file_path):
        return True
    
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        logger.debug(f"🔄 Restaurado: {os.path.basename(file_path)}")
        return True
    
    return False
>>>>>>> Stashed changes


def find_hdr_file(hsi_root: str, folder_name: str) -> Optional[str]:
    """
    Localiza o arquivo HDR em diferentes estruturas de diretório.
    
    Args:
        hsi_root: Diretório raiz das imagens HSI
        folder_name: Nome da pasta da amostra
        
    Returns:
        Caminho do arquivo HDR ou None se não encontrado
    """
    # Possíveis localizações do arquivo HDR
    possible_paths = [
        os.path.join(hsi_root, folder_name, "capture", f"{folder_name}.hdr"),
        os.path.join(hsi_root, folder_name, f"{folder_name}.hdr"),
        os.path.join(hsi_root, folder_name, "capture", "*.hdr"),
    ]
    
    for path in possible_paths:
        if "*" in path:
            matches = glob.glob(path)
            if matches:
                return matches[0]
        elif os.path.exists(path):
            return path
    
    return None


def clean_intermediate_files(output_dir: str, folder_name: str, patterns: List[str]) -> None:
    """
    Remove arquivos intermediários após processamento.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome da amostra
        patterns: Lista de sufixos para remover (ex: ["_roi.csv", "_msc.csv"])
    """
    if config.KEEP_INTERMEDIATE:
        return
    
    for pattern in patterns:
        file_path = os.path.join(output_dir, f"{folder_name}{pattern}")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"🧹 Removido: {os.path.basename(file_path)}")


# ============================================================================
# 📊 FUNÇÕES DO PIPELINE
# ============================================================================

def process_single_sample(folder: str) -> Tuple[bool, str]:
    """
    Processa uma única amostra HSI através de todo o pipeline ETL.
    
    Args:
        folder: Nome da pasta da amostra
        
    Returns:
        Tupla (sucesso: bool, mensagem: str)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"📂 PROCESSANDO: {folder}")
    logger.info(f"{'='*70}\n")
    
    try:
        # Verificar arquivo HDR
        hdr_path = find_hdr_file(config.HSI_ROOT, folder)
        if not hdr_path:
            return False, f"Arquivo HDR não encontrado para {folder}"
        
        logger.info(f"   HDR encontrado: {hdr_path}\n")
        
        # =====================================================
        # 📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas
        # =====================================================
        logger.info("📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas")
        pre_processing_image.run(config.HSI_ROOT, folder)
        
        # Mover arquivo ROI para diretório de saída
        src_roi = os.path.join(config.HSI_ROOT, f"{folder}_roi_filtrado.csv")
        dst_roi = os.path.join(config.OUTPUT_BASE, f"{folder}_roi.csv")
        
        if not os.path.exists(src_roi):
            return False, "Arquivo ROI não gerado"
        
        shutil.move(src_roi, dst_roi)
        logger.info(f"   📝 Movido ROI para: {dst_roi}")
        
        if not csv_has_data(dst_roi):
            if os.path.exists(dst_roi):
                os.remove(dst_roi)
            return False, "ROI sem dados válidos"
        
        # =====================================================
        # 📍 ETAPA 2: Multiplicative Scatter Correction (MSC)
        # =====================================================
        logger.info("\n📍 ETAPA 2: Multiplicative Scatter Correction (MSC)")
        msc_signal.run(config.OUTPUT_BASE, folder)
        
        msc_file = os.path.join(config.OUTPUT_BASE, f"{folder}_msc_signal.csv")
        if not csv_has_data(msc_file):
            return False, "MSC não gerou dados válidos"
        
        # =====================================================
        # 📍 ETAPA 3: Transformações espectrais
        # =====================================================
        logger.info("\n📍 ETAPA 3: Transformações espectrais")
        
        # 3.1 - Log Transform
        logger.info("   3.1 - Transformada Log(1/R)")
        log_transform.run(config.OUTPUT_BASE, folder)
        
        log_file = os.path.join(config.OUTPUT_BASE, f"{folder}_log.csv")
        if not csv_has_data(log_file):
            return False, "Log transform não gerou dados válidos"
        
        # 3.2 - Derivada Savitzky-Golay
        logger.info("   3.2 - Derivada Savitzky-Golay (1ª ordem)")
        derivative_transform.run(config.OUTPUT_BASE, folder)
        
        deriv_file = os.path.join(config.OUTPUT_BASE, f"{folder}_derivada.csv")
        if not csv_has_data(deriv_file):
            return False, "Derivada não gerou dados válidos"
        
        # =====================================================
        # 📍 ETAPA 4: Extração de features
        # =====================================================
        logger.info("\n📍 ETAPA 4: Extração de features")
        
        # Criar backup do arquivo derivada
        backup_deriv = os.path.join(config.OUTPUT_BASE, f"{folder}_derivada_BACKUP.csv")
        shutil.copy2(deriv_file, backup_deriv)
        logger.debug(f"   💾 Backup criado: {os.path.basename(backup_deriv)}")
        
        # 4.1 - Features descritivas
        logger.info("   4.1 - Estatísticas descritivas")
        ensure_file_exists(config.OUTPUT_BASE, folder, "_derivada.csv", backup_deriv)
        descriptive_stats.run(config.OUTPUT_BASE, folder)
        
        # 4.2 - Features espectrais
        logger.info("   4.2 - Features espectrais")
        ensure_file_exists(config.OUTPUT_BASE, folder, "_derivada.csv", backup_deriv)
        spectral_features.run(config.OUTPUT_BASE, folder)
        
        # 4.3 - Índices espectrais SWIR
        logger.info("   4.3 - Índices espectrais SWIR")
        ensure_file_exists(config.OUTPUT_BASE, folder, "_derivada.csv", backup_deriv)
        index_features.run(config.OUTPUT_BASE, folder)
        
        # Remover backup
        if os.path.exists(backup_deriv):
            os.remove(backup_deriv)
            logger.debug("   🧹 Backup removido")
        
        # 4.4 - Combinar features
        logger.info("   4.4 - Combinando features")
        combine_features.run(config.OUTPUT_BASE, folder, remove_inputs=True, move_outputs=False)
        
        logger.info(f"\n✅ Finalizado com sucesso: {folder}")
        return True, "Sucesso"
        
    except Exception as e:
        logger.exception(f"❌ Erro ao processar {folder}")
        return False, str(e)


def combine_all_features(output_dir: str, filename: str = "dataset_final.csv") -> str:
    """
    Combina todos os arquivos de features em um único dataset.
    
    Args:
        output_dir: Diretório com arquivos de features
        filename: Nome do arquivo de saída
        
    Returns:
        Caminho do dataset combinado
    """
    logger.info("Combinando arquivos de features em dataset único...")
    
    # Buscar arquivos de features
    feature_files = glob.glob(os.path.join(output_dir, "*_features.csv"))
    
    if not feature_files:
        raise FileNotFoundError(f"Nenhum arquivo *_features.csv encontrado em: {output_dir}")
    
    logger.info(f"   Encontrados {len(feature_files)} arquivos de features")
    
    # Carregar e combinar DataFrames
    dfs = []
    for f in feature_files:
        df = pd.read_csv(f)
        sample_name = os.path.basename(f).replace("_features.csv", "")
        df["filename"] = sample_name
        dfs.append(df)
        logger.info(f"   ✅ {sample_name}: {len(df)} amostras, {len(df.columns)} features")
    
    # Concatenar
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Salvar
    output_path = os.path.join(output_dir, filename)
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Dataset combinado salvo em: {output_path}")
    logger.info(f"   Total: {len(combined_df)} amostras, {len(combined_df.columns)} colunas")
=======
>>>>>>> Stashed changes
    
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f"pipeline_{timestamp}.log")
    
    # Configurar logger
    logger = logging.getLogger("HSI_Pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicação de handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"📝 Log salvo em: {log_file}")
    
    return logger


<<<<<<< Updated upstream
def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera labels para o dataset baseado nas amostras ATCC.
=======
# Logger global
logger = setup_logging()


# ============================================================================
# 🛠️ FUNÇÕES UTILITÁRIAS
# ============================================================================

def csv_has_data(filepath: str) -> bool:
    """
    Verifica se um arquivo CSV existe e contém dados válidos.
    
    Args:
        filepath: Caminho do arquivo CSV
        
    Returns:
        True se arquivo existe e tem dados, False caso contrário
    """
    if not os.path.exists(filepath):
        return False
    
    if os.path.getsize(filepath) == 0:
        return False
>>>>>>> Stashed changes
    
    Args:
        df: DataFrame com features
        
    Returns:
        DataFrame com coluna 'label' adicionada
    """
    try:
<<<<<<< Updated upstream
        df = label_by_atcc.gerar_labels_por_atcc(df, limiar_sim=config.LIMIAR_SIMILARIDADE)
    except Exception as e:
        logger.warning(f"⚠️ Erro ao gerar labels ATCC: {e}")
        logger.info("   Criando labels padrão (0) para todas as amostras...")
        df["label"] = 0
    
    return df


=======
        df = pd.read_csv(filepath)
        return not df.empty and len(df.columns) > 0
    except Exception as e:
        logger.warning(f"Erro ao ler {filepath}: {e}")
        return False


def ensure_file_exists(
    output_dir: str, 
    folder_name: str, 
    suffix: str, 
    backup_path: Optional[str] = None
) -> bool:
    """
    Garante que um arquivo existe, restaurando do backup se necessário.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome da amostra
        suffix: Sufixo do arquivo (ex: "_derivada.csv")
        backup_path: Caminho do backup (opcional)
        
    Returns:
        True se arquivo existe ou foi restaurado, False caso contrário
    """
    file_path = os.path.join(output_dir, f"{folder_name}{suffix}")
    
    if os.path.exists(file_path):
        return True
    
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        logger.debug(f"🔄 Restaurado: {os.path.basename(file_path)}")
        return True
    
    return False


def find_hdr_file(hsi_root: str, folder_name: str) -> Optional[str]:
    """
    Localiza o arquivo HDR em diferentes estruturas de diretório.
    
    Args:
        hsi_root: Diretório raiz das imagens HSI
        folder_name: Nome da pasta da amostra
        
    Returns:
        Caminho do arquivo HDR ou None se não encontrado
    """
    # Possíveis localizações do arquivo HDR
    possible_paths = [
        os.path.join(hsi_root, folder_name, "capture", f"{folder_name}.hdr"),
        os.path.join(hsi_root, folder_name, f"{folder_name}.hdr"),
        os.path.join(hsi_root, folder_name, "capture", "*.hdr"),
    ]
    
    for path in possible_paths:
        if "*" in path:
            matches = glob.glob(path)
            if matches:
                return matches[0]
        elif os.path.exists(path):
            return path
    
    return None


def clean_intermediate_files(output_dir: str, folder_name: str, patterns: List[str]) -> None:
    """
    Remove arquivos intermediários após processamento.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome da amostra
        patterns: Lista de sufixos para remover (ex: ["_roi.csv", "_msc.csv"])
    """
    if config.KEEP_INTERMEDIATE:
        return
    
    for pattern in patterns:
        file_path = os.path.join(output_dir, f"{folder_name}{pattern}")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"🧹 Removido: {os.path.basename(file_path)}")


# ============================================================================
# 📊 FUNÇÕES DO PIPELINE
# ============================================================================

def process_single_sample(folder: str) -> Tuple[bool, str]:
    """
    Processa uma única amostra HSI através de todo o pipeline ETL.
    
    Args:
        folder: Nome da pasta da amostra
        
    Returns:
        Tupla (sucesso: bool, mensagem: str)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"📂 PROCESSANDO: {folder}")
    logger.info(f"{'='*70}\n")
    
    try:
        # Verificar arquivo HDR
        hdr_path = find_hdr_file(config.HSI_ROOT, folder)
        if not hdr_path:
            return False, f"Arquivo HDR não encontrado para {folder}"
        
        logger.info(f"   HDR encontrado: {hdr_path}\n")
        
        # =====================================================
        # 📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas
        # =====================================================
        logger.info("📍 ETAPA 1: Detecção de ROI e remoção de bandas ruidosas")
        pre_processing_image.run(config.HSI_ROOT, folder)
        
        # Mover arquivo ROI para diretório de saída
        src_roi = os.path.join(config.HSI_ROOT, f"{folder}_roi_filtrado.csv")
        dst_roi = os.path.join(config.OUTPUT_BASE, f"{folder}_roi.csv")
        
        if not os.path.exists(src_roi):
            return False, "Arquivo ROI não gerado"
        
        shutil.move(src_roi, dst_roi)
        logger.info(f"   📝 Movido ROI para: {dst_roi}")
        
        if not csv_has_data(dst_roi):
            if os.path.exists(dst_roi):
                os.remove(dst_roi)
            return False, "ROI sem dados válidos"
        
        # =====================================================
        # 📍 ETAPA 2: Multiplicative Scatter Correction (MSC)
        # =====================================================
        logger.info("\n📍 ETAPA 2: Multiplicative Scatter Correction (MSC)")
        msc_signal.run(config.OUTPUT_BASE, folder)
        
        msc_file = os.path.join(config.OUTPUT_BASE, f"{folder}_msc_signal.csv")
        if not csv_has_data(msc_file):
            return False, "MSC não gerou dados válidos"
        
        # =====================================================
        # 📍 ETAPA 3: Transformações espectrais
        # =====================================================
        logger.info("\n📍 ETAPA 3: Transformações espectrais")
        
        # 3.1 - Log Transform
        logger.info("   3.1 - Transformada Log(1/R)")
        log_transform.run(config.OUTPUT_BASE, folder)
        
        log_file = os.path.join(config.OUTPUT_BASE, f"{folder}_log.csv")
        if not csv_has_data(log_file):
            return False, "Log transform não gerou dados válidos"
        
        # 3.2 - Derivada Savitzky-Golay
        logger.info("   3.2 - Derivada Savitzky-Golay (1ª ordem)")
        derivative_transform.run(config.OUTPUT_BASE, folder)
        
        deriv_file = os.path.join(config.OUTPUT_BASE, f"{folder}_derivada.csv")
        if not csv_has_data(deriv_file):
            return False, "Derivada não gerou dados válidos"
        
        # =====================================================
        # 📍 ETAPA 4: Extração de features
        # =====================================================
        logger.info("\n📍 ETAPA 4: Extração de features")
        
        # Criar backup do arquivo derivada
        backup_deriv = os.path.join(config.OUTPUT_BASE, f"{folder}_derivada_BACKUP.csv")
        shutil.copy2(deriv_file, backup_deriv)
        logger.debug(f"   💾 Backup criado: {os.path.basename(backup_deriv)}")
        
        # 4.1 - Features descritivas
        logger.info("   4.1 - Estatísticas descritivas")
        ensure_file_exists(config.OUTPUT_BASE, folder, "_derivada.csv", backup_deriv)
        descriptive_stats.run(config.OUTPUT_BASE, folder)
        
        # 4.2 - Features espectrais
        logger.info("   4.2 - Features espectrais")
        ensure_file_exists(config.OUTPUT_BASE, folder, "_derivada.csv", backup_deriv)
        spectral_features.run(config.OUTPUT_BASE, folder)
        
        # 4.3 - Índices espectrais SWIR
        logger.info("   4.3 - Índices espectrais SWIR")
        ensure_file_exists(config.OUTPUT_BASE, folder, "_derivada.csv", backup_deriv)
        index_features.run(config.OUTPUT_BASE, folder)
        
        # Remover backup
        if os.path.exists(backup_deriv):
            os.remove(backup_deriv)
            logger.debug("   🧹 Backup removido")
        
        # 4.4 - Combinar features
        logger.info("   4.4 - Combinando features")
        combine_features.run(config.OUTPUT_BASE, folder, remove_inputs=True, move_outputs=False)
        
        logger.info(f"\n✅ Finalizado com sucesso: {folder}")
        return True, "Sucesso"
        
    except Exception as e:
        logger.exception(f"❌ Erro ao processar {folder}")
        return False, str(e)


def combine_all_features(output_dir: str, filename: str = "dataset_final.csv") -> str:
    """
    Combina todos os arquivos de features em um único dataset.
    
    Args:
        output_dir: Diretório com arquivos de features
        filename: Nome do arquivo de saída
        
    Returns:
        Caminho do dataset combinado
    """
    logger.info("Combinando arquivos de features em dataset único...")
    
    # Buscar arquivos de features
    feature_files = glob.glob(os.path.join(output_dir, "*_features.csv"))
    
    if not feature_files:
        raise FileNotFoundError(f"Nenhum arquivo *_features.csv encontrado em: {output_dir}")
    
    logger.info(f"   Encontrados {len(feature_files)} arquivos de features")
    
    # Carregar e combinar DataFrames
    dfs = []
    for f in feature_files:
        df = pd.read_csv(f)
        sample_name = os.path.basename(f).replace("_features.csv", "")
        df["filename"] = sample_name
        dfs.append(df)
        logger.info(f"   ✅ {sample_name}: {len(df)} amostras, {len(df.columns)} features")
    
    # Concatenar
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Salvar
    output_path = os.path.join(output_dir, filename)
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Dataset combinado salvo em: {output_path}")
    logger.info(f"   Total: {len(combined_df)} amostras, {len(combined_df.columns)} colunas")
    
    return output_path


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera labels para o dataset baseado nas amostras ATCC.
    
    Args:
        df: DataFrame com features
        
    Returns:
        DataFrame com coluna 'label' adicionada
    """
    try:
        df = label_by_atcc.gerar_labels_por_atcc(df, limiar_sim=config.LIMIAR_SIMILARIDADE)
    except Exception as e:
        logger.warning(f"⚠️ Erro ao gerar labels ATCC: {e}")
        logger.info("   Criando labels padrão (0) para todas as amostras...")
        df["label"] = 0
    
    return df


>>>>>>> Stashed changes
def run_feature_selection(dataset_path: str, features_dir: str) -> None:
    """
    Executa seleção de features em duas etapas: MI e XGBoost.
    
    Args:
        dataset_path: Caminho do dataset com features e labels
        features_dir: Diretório para salvar features selecionadas
    """
    df = pd.read_csv(dataset_path)
    
    # Verificar variabilidade das labels
    if df['label'].nunique() < 2:
        logger.warning("⚠️ Labels sem variabilidade! Pulando seleção de features...")
        logger.warning("   (Todas as amostras têm o mesmo label)")
        return
    
    # Etapa 1: Mutual Information
    logger.info("7.1 - Seleção via Mutual Information...")
    feature_selection_mi.run(dataset_path, features_dir)
    
    # Etapa 2: XGBoost Feature Importance
    logger.info("\n7.2 - Seleção via XGBoost feature importance...")
    mi_features_path = os.path.join(features_dir, "features_selected_mi.csv")
    
    if os.path.exists(mi_features_path):
        feature_selection_xgboost.run(dataset_path, mi_features_path, features_dir)
    else:
        logger.warning(f"   ⚠️ Arquivo MI não encontrado: {mi_features_path}")


# ============================================================================
# 🚀 FUNÇÃO PRINCIPAL
# ============================================================================

def main():
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    pastas = sorted([
        nome for nome in os.listdir(HSI_ROOT)
        if os.path.isdir(os.path.join(HSI_ROOT, nome))
    ])
    print(f"📂 {len(pastas)} imagens encontradas")

    frames_msc   = []
    frames_log   = []
    frames_deriv = []

    for folder_name in pastas:
        print(f"\n🔬 Processando: {folder_name}")

        # ------------------------------------------------------------------
        # ETAPA 0 — Filtro ROI
        # ------------------------------------------------------------------
        csv_roi = os.path.join(HSI_ROOT, f"{folder_name}_roi_filtrado.csv")

        if not roi_valido(csv_roi):
            pre_processing_image(output_dir=HSI_ROOT, folder_name=folder_name)

        if not roi_valido(csv_roi):
            print(f"  ⚠️  ROI inválido ou não detectado. Pulando {folder_name}.")
            continue

        espectros = pd.read_csv(csv_roi).values
        print(f"  ✅ ROI carregado: {espectros.shape}")

        # ------------------------------------------------------------------
        # BRANCH MSC → engenharia de atributos
        # ------------------------------------------------------------------
        df_msc = extrair_features(espectros, msc_signal)
        df_msc.insert(0, "sample_id", folder_name)
        frames_msc.append(df_msc)

        # ------------------------------------------------------------------
        # BRANCH Log(1/R) → engenharia de atributos
        # ------------------------------------------------------------------
        df_log = extrair_features(espectros, log_transform)
        df_log.insert(0, "sample_id", folder_name)
        frames_log.append(df_log)

        # ------------------------------------------------------------------
        # BRANCH Savitzky-Golay → engenharia de atributos
        # ------------------------------------------------------------------
        df_deriv = extrair_features(espectros, derivative_transform)
        df_deriv.insert(0, "sample_id", folder_name)
        frames_deriv.append(df_deriv)

    # ----------------------------------------------------------------------
    # CONSOLIDAÇÃO + ROTULAGEM + SELEÇÃO
    # ----------------------------------------------------------------------
    branches = [
        ("msc",   frames_msc),
        ("log",   frames_log),
        ("deriv", frames_deriv),
    ]

    for nome, frames in branches:
        if not frames:
            print(f"\n⚠️  Nenhuma amostra para o branch {nome.upper()}.")
            continue

        df = pd.concat(frames, ignore_index=True)
        df = label_by_atcc(df)

        print(f"\n{'='*60}")
        print(f"Branch {nome.upper()} — {df.shape[0]} amostras × {df.shape[1]} colunas")
        print(f"Classes: {df['label'].value_counts().to_dict()}")

        aplicar_selecao(df, nome)

    print(f"\n✅ Pipeline concluído — 6 datasets gerados em: {FEATURES_DIR}")
    print("   msc_mi.csv   | msc_xgb.csv")
    print("   log_mi.csv   | log_xgb.csv")
    print("   deriv_mi.csv | deriv_xgb.csv")
=======
    """Executa o pipeline completo de processamento HSI."""
    
    logger.info("🚀 INICIANDO PIPELINE DE PROCESSAMENTO HSI")
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info(f"📁 Configurações:")
    logger.info(f"   Base: {config.BASE_DIR}")
    logger.info(f"   HSI Root: {config.HSI_ROOT}")
    logger.info(f"   Output: {config.OUTPUT_BASE}")
    logger.info(f"   Features: {config.FEATURES_DIR}")
    logger.info("")
    
=======
    """Executa o pipeline completo de processamento HSI."""
    
    logger.info("🚀 INICIANDO PIPELINE DE PROCESSAMENTO HSI")
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info(f"📁 Configurações:")
    logger.info(f"   Base: {config.BASE_DIR}")
    logger.info(f"   HSI Root: {config.HSI_ROOT}")
    logger.info(f"   Output: {config.OUTPUT_BASE}")
    logger.info(f"   Features: {config.FEATURES_DIR}")
    logger.info("")
    
>>>>>>> Stashed changes
    # Criar diretórios necessários
    os.makedirs(config.OUTPUT_BASE, exist_ok=True)
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    
    # Validar diretório de entrada
    if not os.path.exists(config.HSI_ROOT):
        logger.error(f"❌ Diretório HSI não encontrado: {config.HSI_ROOT}")
        sys.exit(1)
    
    # Listar amostras
    folders = [
        f for f in os.listdir(config.HSI_ROOT) 
        if os.path.isdir(os.path.join(config.HSI_ROOT, f))
    ]
    
    if not folders:
        logger.error(f"❌ Nenhuma pasta de amostra encontrada em: {config.HSI_ROOT}")
        sys.exit(1)
    
    logger.info(f"📂 Encontradas {len(folders)} amostras para processar\n")
    
    # =====================================================
    # FASE 1: Processamento individual de cada amostra
    # =====================================================
    processed_count = 0
    error_count = 0
    errors = []
    
    for folder in folders:
        success, message = process_single_sample(folder)
        
        if success:
            processed_count += 1
        else:
            error_count += 1
            errors.append((folder, message))
            logger.warning(f"   ⚠️ {message}")
    
    # Resumo do processamento
    logger.info(f"\n{'='*70}")
    logger.info("📊 RESUMO DO PRÉ-PROCESSAMENTO")
    logger.info(f"{'='*70}")
    logger.info(f"   Total de amostras: {len(folders)}")
    logger.info(f"   Processadas com sucesso: {processed_count}")
    logger.info(f"   Erros: {error_count}")
    
    if errors:
        logger.info("\n   Amostras com erro:")
        for folder, msg in errors:
            logger.info(f"      - {folder}: {msg}")
    
    if processed_count == 0:
        logger.error("\n❌ Nenhuma amostra foi processada com sucesso!")
        sys.exit(1)
    
    # =====================================================
    # FASE 2: Combinação de features
    # =====================================================
    logger.info(f"\n{'='*70}")
    logger.info("📦 ETAPA 5: Combinando features de todas as amostras")
    logger.info(f"{'='*70}\n")
    
    try:
        final_dataset_path = combine_all_features(config.OUTPUT_BASE)
    except FileNotFoundError as e:
        logger.error(f"❌ Erro: {e}")
        sys.exit(1)
    
    # =====================================================
    # FASE 3: Geração de labels
    # =====================================================
    logger.info(f"\n{'='*70}")
    logger.info("🏷️ ETAPA 6: Gerando labels (ATCC = Resistente)")
    logger.info(f"{'='*70}\n")
    
    df = pd.read_csv(final_dataset_path)
    df = generate_labels(df)
    
    # Limpeza e preparação do dataset final
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if 'label' not in numeric_cols:
        numeric_cols.append('label')
    
    cols_to_keep = ["filename"] + [c for c in numeric_cols if c != "filename"]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    df = df[cols_to_keep].fillna(0)
    df.to_csv(final_dataset_path, index=False)
    
    n_positivos = int(df['label'].sum())
    n_negativos = len(df) - n_positivos
    
    logger.info(f"✅ Dataset final salvo: {final_dataset_path}")
    logger.info(f"   Amostras: {len(df)}")
    logger.info(f"   Features: {len(df.columns) - 2}")
    logger.info(f"   Labels positivos (resistentes): {n_positivos}")
    logger.info(f"   Labels negativos (sensíveis): {n_negativos}")
    
    # =====================================================
    # FASE 4: Seleção de features
    # =====================================================
    logger.info(f"\n{'='*70}")
    logger.info("📊 ETAPA 7: Seleção de features")
    logger.info(f"{'='*70}\n")
    
    run_feature_selection(final_dataset_path, config.FEATURES_DIR)
    
    # =====================================================
    # FINALIZAÇÃO
    # =====================================================
    logger.info(f"\n{'='*70}")
    logger.info("🏁 PIPELINE ETL FINALIZADO COM SUCESSO!")
    logger.info(f"{'='*70}")
    logger.info("")
    logger.info("📁 Arquivos gerados:")
    logger.info(f"   - Dataset final: {final_dataset_path}")
    logger.info(f"   - Features MI: {config.FEATURES_DIR}/features_selected_mi.csv")
    logger.info(f"   - Features XGBoost: {config.FEATURES_DIR}/features_selected_xgboost.csv")
    logger.info("")
    logger.info("🚀 Próximos passos:")
    logger.info("   1. python ml/split_dataset.py  → Dividir treino/teste")
    logger.info("   2. python ml/train.py          → Treinar modelo com Grid Search")
    logger.info("   3. python ml/evaluate.py       → Avaliar métricas")
    logger.info("")
    logger.info(f"⏱️ Finalizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()