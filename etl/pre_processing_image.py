import os
import numpy as np
import cv2
import spectral
import pandas as pd

KEEP_INTERMEDIATE = False  # ❌ Remove arquivos intermediários após processamento

def run(output_dir: str, folder_name: str):
    """
    Processa imagem HSI: detecta ROI, remove bandas ruidosas e salva CSV filtrado.
    
    Args:
        output_dir: Diretório de saída
        folder_name: Nome base do arquivo (ex: "351_1_240506-160920")
    """
    
    # Construir caminho do arquivo HDR
    input_path = os.path.join(
        output_dir, 
        folder_name, 
        "capture", 
        f"{folder_name}.hdr"
    )
    output_path = os.path.join(output_dir, f"{folder_name}_roi_filtrado.csv")
    
    # Verificar se arquivo existe
    if not os.path.exists(input_path):
        print(f"{input_path} não encontrado. Criando saída vazia {output_path}")
        pd.DataFrame().to_csv(output_path, index=False)
        return
    
    # ==============================================================================
    # 1. CARREGAMENTO DA IMAGEM HSI
    # ==============================================================================
    
    try:
        img = spectral.open_image(input_path).load()
        print(f"✅ Imagem carregada: {img.shape}")
    except Exception as e:
        print(f"❌ Erro ao carregar {input_path}: {e}")
        pd.DataFrame().to_csv(output_path, index=False)
        return
    
    # ==============================================================================
    # 2. DETECÇÃO AUTOMÁTICA DO ROI CIRCULAR
    # ==============================================================================
    
    altura, largura, n_bandas = img.shape
    b_idx = [min(50, n_bandas-1), min(30, n_bandas-1), min(10, n_bandas-1)]
    rgb = spectral.get_rgb(img, b_idx)
    gray = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    
    min_dim = min(altura, largura)
    min_r = max(10, int(min_dim * 0.15))
    max_r = int(min_dim * 0.48)
    
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(min_dim * 0.2)),
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r
    )
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    if circles is not None:
        x, y, r = circles[0][0]
        x, y, r = int(x), int(y), int(r)
        cv2.circle(mask, (x, y), r, 1, thickness=-1)
        print(f"✅ ROI detectado (Hough): centro=({x}, {y}), raio={r}")
    else:
        # Fallback 1: Otsu Thresholding + Enclosing Circle
        _, thresh = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            if radius >= min_r:
                x, y, r = int(cx), int(cy), int(radius)
                cv2.circle(mask, (x, y), r, 1, thickness=-1)
                print(f"✅ ROI detectado (Contorno/Otsu): centro=({x}, {y}), raio={r}")
            else:
                x, y, r = largura // 2, altura // 2, int(min_dim * 0.35)
                cv2.circle(mask, (x, y), r, 1, thickness=-1)
                print(f"⚠️ ROI padrão central aplicado: centro=({x}, {y}), raio={r}")
        else:
            x, y, r = largura // 2, altura // 2, int(min_dim * 0.35)
            cv2.circle(mask, (x, y), r, 1, thickness=-1)
            print(f"⚠️ ROI padrão central aplicado: centro=({x}, {y}), raio={r}")
    
    # ==============================================================================
    # 3. RECORTE E REMOÇÃO DO FUNDO
    # ==============================================================================
    
    altura, largura, n_bandas = img.shape
    x, y, r = int(x), int(y), int(r)
    
    x_min = max(0, x - r)
    x_max = min(largura, x + r)
    y_min = max(0, y - r)
    y_max = min(altura, y + r)
    
    roi_cropped = img[y_min:y_max, x_min:x_max, :]
    mask_cropped = mask[y_min:y_max, x_min:x_max]
    
    # Transformar ROI em matriz 2D (pixels x bandas)
    altura_crop, largura_crop, _ = roi_cropped.shape
    roi_reshaped = roi_cropped.reshape(-1, n_bandas)
    
    # Extrair apenas pixels válidos (dentro do círculo)
    mask_flat = mask_cropped.flatten()
    roi_pixels = roi_reshaped[mask_flat > 0]
    
    if roi_pixels.shape[0] == 0:
        print(f"⚠️ Nenhum pixel válido no ROI. Criando saída vazia.")
        pd.DataFrame().to_csv(output_path, index=False)
        return
    
    print(f"✅ Pixels válidos no ROI: {roi_pixels.shape[0]}")
    
    # ==============================================================================
    # 4. CÁLCULO DE MÉTRICAS (SNR, VARIÂNCIA)
    # ==============================================================================
    
    snr_list = []
    variance_list = []
    mean_list = []
    
    for banda in range(n_bandas):
        banda_data = roi_pixels[:, banda]
        
        media = np.mean(banda_data)
        variancia = np.var(banda_data)
        std = np.std(banda_data)
        snr = media / std if std > 0 else 0
        
        snr_list.append(snr)
        variance_list.append(variancia)
        mean_list.append(media)
    
    snr_array = np.array(snr_list)
    variance_array = np.array(variance_list)
    
    # ==============================================================================
    # 5. REMOÇÃO DE BANDAS RUIDOSAS
    # ==============================================================================
    
    # Critério 1: SNR muito baixo (abaixo do percentil 15)
    snr_threshold = np.percentile(snr_array, 15)
    bandas_baixo_snr = np.where(snr_array < snr_threshold)[0]
    
    # Critério 2: Variância muito baixa (bandas quase constantes)
    var_threshold = np.percentile(variance_array, 5)
    bandas_baixa_variancia = np.where(variance_array < var_threshold)[0]
    
    # Critério 3: Bandas nas extremidades espectrais
    n_remover_extremos = 15
    bandas_inicio = np.arange(0, n_remover_extremos)
    bandas_fim = np.arange(n_bandas - n_remover_extremos, n_bandas)
    
    # Combinar todas as bandas a serem removidas
    bandas_remover = np.unique(np.concatenate([
        bandas_baixo_snr,
        bandas_baixa_variancia,
        bandas_inicio,
        bandas_fim
    ]))
    
    bandas_manter = np.setdiff1d(np.arange(n_bandas), bandas_remover)
    
    print(f"✅ Bandas originais: {n_bandas}")
    print(f"✅ Bandas removidas: {len(bandas_remover)}")
    print(f"✅ Bandas mantidas: {len(bandas_manter)} ({(len(bandas_manter)/n_bandas)*100:.1f}%)")
    
    # Aplicar filtragem
    roi_filtrado = roi_cropped[:, :, bandas_manter]
    
    # ==============================================================================
    # 6. SALVAMENTO EM CSV (APENAS PIXELS)
    # ==============================================================================
    
    altura_final, largura_final, n_bandas_final = roi_filtrado.shape
    roi_filtrado_reshaped = roi_filtrado.reshape(-1, n_bandas_final)
    roi_filtrado_pixels = roi_filtrado_reshaped[mask_flat > 0]
    
    # Criar DataFrame apenas com valores espectrais
    colunas_bandas = [f'banda_{i}' for i in bandas_manter]
    df_roi_filtrado = pd.DataFrame(roi_filtrado_pixels, columns=colunas_bandas)
    
    # Salvar CSV
    df_roi_filtrado.to_csv(output_path, index=False)
    print(f"✅ ROI filtrado salvo: {output_path}")
    print(f"   Dimensões: {df_roi_filtrado.shape[0]} pixels × {df_roi_filtrado.shape[1]} bandas")
    
    # Limpar arquivos intermediários se necessário
    if not KEEP_INTERMEDIATE:
        # Aqui você pode adicionar lógica para remover arquivos temporários
        print(f"🧹 Modo limpeza: KEEP_INTERMEDIATE={KEEP_INTERMEDIATE}")


if __name__ == "__main__":
    base_dir = os.getenv("IC2025_BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    output_dir = os.path.join(base_dir, "dataSet", "hsi_original")
    folder_name = "351_1_240506-160920"

    run(output_dir, folder_name)