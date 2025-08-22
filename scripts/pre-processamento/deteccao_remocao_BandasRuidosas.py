import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from typing import Dict, List, Tuple, Optional

# Configuração de caminhos
input_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_sem_bandas_ruidosas.csv"
output_csv = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed/dados_com_atributos_espectrais.csv"

def extrair_estatisticas_descritivas(espectro: np.ndarray) -> Dict[str, float]:
    """
    Extrai estatísticas descritivas do espectro
    
    Args:
        espectro: Array 1D com valores espectrais
        
    Returns:
        Dicionário com estatísticas descritivas
    """
    atributos = {
        'media': np.mean(espectro),
        'desvio_padrao': np.std(espectro),
        'variancia': np.var(espectro),
        'mediana': np.median(espectro),
        'minimo': np.min(espectro),
        'maximo': np.max(espectro),
        'amplitude': np.ptp(espectro),
        'assimetria': stats.skew(espectro),
        'curtose': stats.kurtosis(espectro),
        'q25': np.percentile(espectro, 25),
        'q75': np.percentile(espectro, 75),
        'intervalo_interquartil': np.percentile(espectro, 75) - np.percentile(espectro, 25)
    }
    
    return atributos

def calcular_segunda_derivada(espectro: np.ndarray, tamanho_janela: int = 5, ordem_polinomio: int = 2) -> np.ndarray:
    """
    Calcula segunda derivada usando filtro Savitzky-Golay
    
    Args:
        espectro: Array 1D com valores espectrais
        tamanho_janela: Tamanho da janela para suavização
        ordem_polinomio: Ordem do polinômio para ajuste
        
    Returns:
        Array com segunda derivada
    """
    segunda_derivada = savgol_filter(espectro, tamanho_janela, ordem_polinomio, deriv=2)
    return segunda_derivada

def extrair_atributos_derivadas(espectro: np.ndarray) -> Dict[str, float]:
    """
    Extrai características da segunda derivada
    
    Args:
        espectro: Array 1D com valores espectrais
        
    Returns:
        Dicionário com características da segunda derivada
    """
    atributos = {}
    
    # Características da segunda derivada
    seg_deriv = calcular_segunda_derivada(espectro)
    atributos.update({
        'seg_deriv_media': np.mean(seg_deriv),
        'seg_deriv_desvio': np.std(seg_deriv),
        'seg_deriv_max': np.max(seg_deriv),
        'seg_deriv_min': np.min(seg_deriv),
        'seg_deriv_amplitude': np.ptp(seg_deriv),
        'seg_deriv_assimetria': stats.skew(seg_deriv),
        'seg_deriv_curtose': stats.kurtosis(seg_deriv)
    })
    
    return atributos

def calcular_razoes_bandas(espectro: np.ndarray, faixas_bandas: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Calcula razões entre bandas espectrais específicas
    
    Args:
        espectro: Array 1D com valores espectrais
        faixas_bandas: Lista de tuplas com índices (início, fim) das bandas
        
    Returns:
        Dicionário com razões entre bandas
    """
    atributos = {}
    
    # Calcula média de cada banda
    medias_bandas = []
    for i, (inicio, fim) in enumerate(faixas_bandas):
        media_banda = np.mean(espectro[inicio:fim+1])
        medias_bandas.append(media_banda)
        atributos[f'banda_{i+1}_media'] = media_banda
        atributos[f'banda_{i+1}_max'] = np.max(espectro[inicio:fim+1])
        atributos[f'banda_{i+1}_min'] = np.min(espectro[inicio:fim+1])
        atributos[f'banda_{i+1}_desvio'] = np.std(espectro[inicio:fim+1])
    
    # Calcula razões entre todas as combinações de bandas
    for i in range(len(medias_bandas)):
        for j in range(i+1, len(medias_bandas)):
            if medias_bandas[j] != 0:  # Evita divisão por zero
                razao = medias_bandas[i] / medias_bandas[j]
                atributos[f'razao_banda_{i+1}_banda_{j+1}'] = razao
                
                # Diferença normalizada
                diff_norm = (medias_bandas[i] - medias_bandas[j]) / (medias_bandas[i] + medias_bandas[j])
                atributos[f'diff_norm_banda_{i+1}_banda_{j+1}'] = diff_norm
    
    return atributos

def calcular_indices_espectrais(espectro: np.ndarray, comprimentos_onda: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula índices espectrais derivados empiricamente
    
    Args:
        espectro: Array 1D com valores espectrais
        comprimentos_onda: Array com comprimentos de onda correspondentes
        
    Returns:
        Dicionário com índices espectrais
    """
    atributos = {}
    
    # Área sob a curva
    atributos['area_sob_curva'] = np.trapz(espectro)
    
    # Centro de massa espectral
    if comprimentos_onda is not None and len(comprimentos_onda) == len(espectro):
        soma_ponderada = np.sum(espectro * comprimentos_onda)
        soma_total = np.sum(espectro)
        if soma_total != 0:
            atributos['centroide_espectral'] = soma_ponderada / soma_total
    
    # Índice de uniformidade
    media_val = np.mean(espectro)
    if media_val != 0:
        atributos['indice_uniformidade'] = np.std(espectro) / media_val
    
    # Assimetria normalizada
    desvio = np.std(espectro)
    if desvio != 0:
        atributos['assimetria_normalizada'] = stats.skew(espectro) / (desvio ** 1.5)
    else:
        atributos['assimetria_normalizada'] = 0
    
    # Detecção de picos
    picos_indices, propriedades_picos = find_peaks(espectro, height=np.mean(espectro))
    atributos['num_picos'] = len(picos_indices)
    
    if len(picos_indices) > 0:
        atributos['intensidade_max_pico'] = np.max(espectro[picos_indices])
        atributos['intensidade_media_picos'] = np.mean(espectro[picos_indices])
        atributos['posicao_pico_max'] = picos_indices[np.argmax(espectro[picos_indices])]
    else:
        atributos['intensidade_max_pico'] = 0
        atributos['intensidade_media_picos'] = 0
        atributos['posicao_pico_max'] = 0
    
    # Índices de forma espectral
    atributos['energia_espectral'] = np.sum(espectro ** 2)
    atributos['entropia_espectral'] = -np.sum(espectro * np.log(espectro + 1e-10))  # Evita log(0)
    
    # Momentos espectrais
    momentos = []
    for ordem in range(1, 5):  # 1° ao 4° momento
        momento = np.sum((np.arange(len(espectro)) ** ordem) * espectro) / np.sum(espectro)
        momentos.append(momento)
        atributos[f'momento_{ordem}'] = momento
    
    return atributos

def definir_faixas_bandas_padrao(tamanho_espectro: int) -> List[Tuple[int, int]]:
    """
    Define faixas de bandas padrão baseadas no tamanho do espectro
    
    Args:
        tamanho_espectro: Número de pontos no espectro
        
    Returns:
        Lista de tuplas com faixas de bandas
    """
    # Divide o espectro em 6 regiões aproximadamente iguais
    tamanho_banda = tamanho_espectro // 6
    faixas = []
    
    for i in range(6):
        inicio = i * tamanho_banda
        fim = min((i + 1) * tamanho_banda - 1, tamanho_espectro - 1)
        if inicio < tamanho_espectro:
            faixas.append((inicio, fim))
    
    return faixas

def extrair_atributos_espectrais(input_path: str, output_path: str, faixas_bandas_custom: Optional[List[Tuple[int, int]]] = None):
    """
    Extrai atributos espectrais completos dos dados
    
    Args:
        input_path: Caminho do arquivo CSV de entrada
        output_path: Caminho do arquivo CSV de saída
        faixas_bandas_custom: Faixas de bandas personalizadas (opcional)
    """
    # Carrega os dados
    df = pd.read_csv(input_path)
    print(f"📊 Carregando dados de: {input_path}")
    print(f"📏 Dimensões dos dados: {df.shape}")
    
    # Separa dados espectrais de outras colunas (assumindo que as primeiras colunas são metadados)
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    dados_espectrais = df[colunas_numericas].values
    
    # Define faixas de bandas
    if faixas_bandas_custom is None:
        faixas_bandas = definir_faixas_bandas_padrao(dados_espectrais.shape[1])
    else:
        faixas_bandas = faixas_bandas_custom
    
    print(f"🎯 Faixas de bandas definidas: {len(faixas_bandas)} bandas")
    
    # Lista para armazenar todos os atributos
    todos_atributos = []
    
    print("🔄 Extraindo atributos espectrais...")
    
    # Processa cada espectro
    for i, espectro in enumerate(dados_espectrais):
        if i % 100 == 0:
            print(f"   Processando espectro {i+1}/{len(dados_espectrais)}")
        
        atributos_linha = {}
        
        # Estatísticas descritivas
        stats_desc = extrair_estatisticas_descritivas(espectro)
        atributos_linha.update(stats_desc)
        
        # Atributos das derivadas
        attrs_deriv = extrair_atributos_derivadas(espectro)
        atributos_linha.update(attrs_deriv)
        
        # Razões entre bandas
        razoes_bandas = calcular_razoes_bandas(espectro, faixas_bandas)
        atributos_linha.update(razoes_bandas)
        
        # Índices espectrais
        indices_spec = calcular_indices_espectrais(espectro)
        atributos_linha.update(indices_spec)
        
        todos_atributos.append(atributos_linha)
    
    # Cria DataFrame com os atributos
    df_atributos = pd.DataFrame(todos_atributos)
    
    # Adiciona colunas não-numéricas originais se existirem
    colunas_nao_numericas = df.select_dtypes(exclude=[np.number]).columns
    if len(colunas_nao_numericas) > 0:
        df_final = pd.concat([df[colunas_nao_numericas].reset_index(drop=True), 
                             df_atributos.reset_index(drop=True)], axis=1)
    else:
        df_final = df_atributos
    
    # Salva o resultado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ Atributos espectrais extraídos com sucesso!")
    print(f"📊 Total de atributos criados: {len(df_atributos.columns)}")
    print(f"💾 Arquivo salvo em: {output_path}")
    
    # Mostra resumo dos atributos
    print("\n📋 Resumo dos tipos de atributos extraídos:")
    print(f"   • Estatísticas descritivas: 12 atributos")
    print(f"   • Segunda derivada: 7 atributos")
    print(f"   • Bandas espectrais: {len([k for k in df_atributos.columns if 'banda_' in k])} atributos")
    print(f"   • Razões entre bandas: {len([k for k in df_atributos.columns if 'razao_' in k or 'diff_norm_' in k])} atributos")
    print(f"   • Índices espectrais: {len([k for k in df_atributos.columns if k in ['area_sob_curva', 'centroide_espectral', 'indice_uniformidade', 'assimetria_normalizada', 'num_picos', 'intensidade_max_pico', 'intensidade_media_picos', 'posicao_pico_max', 'energia_espectral', 'entropia_espectral'] or 'momento_' in k])} atributos")

# Executa a extração de atributos
if __name__ == "__main__":
    # Exemplo com faixas de bandas personalizadas (opcional)
    # faixas_custom = [(0, 50), (51, 150), (151, 300), (301, 500), (501, 800), (801, 1000)]
    
    extrair_atributos_espectrais(input_csv, output_csv)