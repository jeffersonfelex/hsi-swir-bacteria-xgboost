import os
import pandas as pd
import numpy as np
import spectral as spy
import cv2

def run(hdr_path: str, output_dir: str, folder_name: str):
    output_csv = os.path.join(output_dir, f"{folder_name}_roi.csv")
    output_img = os.path.join(output_dir, f"{folder_name}_roi.png")

    try:
        img = spy.open_image(hdr_path).load()
        img = np.array(img)  # garante array numpy

        print(f"📐 {folder_name} — shape carregada: {img.shape}, ndim={img.ndim}")

        # Se a imagem for 2D, não há bandas espectrais
        if img.ndim == 2:
            print(f"⚠️ {folder_name}: imagem sem dimensão espectral. Pulando.")
            pd.DataFrame().to_csv(output_csv, index=False)
            return

        # Corrige caso esteja em (bandas, altura, largura)
        if img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))
            print(f"🔄 Transposto para {img.shape}")

        # Gera imagem RGB para detectar ROI
        bands = img.shape[2]
        rgb = spy.get_rgb(img, [min(50, bands-1), min(30, bands-1), min(10, bands-1)])
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 2)

        # Detecta círculo da ROI
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=50,
            maxRadius=200
        )

        mask = np.zeros(gray.shape, dtype=np.uint8)
        if circles is not None:
            x, y, r = circles[0][0]
            cv2.circle(mask, (int(x), int(y)), int(r), 1, thickness=-1)

        # Extrai pixels da ROI (todos os pixels da máscara)
        roi_pixels = img[mask == 1]

        if roi_pixels.size == 0:
            print(f"⚠️ ROI vazia em {folder_name}. Criando arquivo vazio.")
            pd.DataFrame().to_csv(output_csv, index=False)
            return

        # Salva espectros dos pixels
        pd.DataFrame(roi_pixels).to_csv(output_csv, index=False)

        # Salva imagem da ROI para visualização
        roi_rgb = spy.get_rgb(img * mask[:, :, np.newaxis], [min(50, bands-1), min(30, bands-1), min(10, bands-1)])
        roi_rgb_norm = (roi_rgb / roi_rgb.max()) if roi_rgb.max() > 0 else roi_rgb
        cv2.imwrite(output_img, (roi_rgb_norm * 255).astype(np.uint8))

        print(f"✅ ROI extraída ({roi_pixels.shape[0]} pixels, {roi_pixels.shape[1]} bandas): {output_csv}")

    except Exception as e:
        print(f"❌ Erro ao extrair ROI de {folder_name}: {e}")
        pd.DataFrame().to_csv(output_csv, index=False)
