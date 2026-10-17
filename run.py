"""
Runner e Orquestrador do Ambiente de Pesquisa IC-2025
====================================================
Classificação espectral de bactérias multirresistentes (SWIR)
com Aprendizado de Máquina (XGBoost, PCA, Spectral Feature Engineering).

Uso:
    python run.py check          -> Verifica ambiente, dependências e GPU
    python run.py init           -> Cria toda a estrutura de diretórios necessária
    python run.py mock           -> Gera dados sintéticos HSI para teste rápido
    python run.py etl            -> Executa pipeline ETL completo, 4 branches (main.py)
    python run.py split          -> Divide datasets em treino/teste por imagem (ml/split_dataset.py)
    python run.py train          -> Treina modelos XGBoost com Grid Search (ml/train.py)
    python run.py evaluate       -> Avalia no conjunto de teste externo por imagem (ml/evaluate.py)
    python run.py all            -> Executa pipeline completo: etl -> split -> train -> evaluate

Observação: o antigo "raw" (build_raw_pipeline.py) e "train-raw" foram removidos —
o branch RAW agora é gerado pelo próprio main.py junto com MSC/Log/Derivada.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import numpy as np

# Garantir codificação UTF-8 no Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parent
os.environ["IC2025_BASE_DIR"] = str(BASE_DIR)
os.environ["PYTHONIOENCODING"] = "utf-8"


def check_env():
    """Verifica ambiente, dependências instaladas e disponibilidade de aceleração."""
    print("=" * 70)
    print("🔍 VERIFICAÇÃO DO AMBIENTE DE PESQUISA (IC-2025)")
    print("=" * 70)
    print(f"📁 Diretório Raiz: {BASE_DIR}")
    print(f"🐍 Python Executável: {sys.executable}")
    print(f"🐍 Versão Python: {sys.version.split()[0]}")
    
    # Checar módulos essenciais
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("xgboost", "XGBoost"),
        ("spectral", "Spectral (ENVI/HSI)"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("lightgbm", "LightGBM"),
        ("catboost", "CatBoost"),
        ("jupyter", "Jupyter"),
    ]
    
    print("\n📦 Pacotes Instalados:")
    all_ok = True
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "OK")
            print(f"   ✅ {label:25s} : {ver}")
        except ImportError:
            print(f"   ❌ {label:25s} : NÃO INSTALADO")
            all_ok = False
            
    # Checar GPU/CUDA no XGBoost
    print("\n⚡ Dispositivo de Treinamento:")
    try:
        from xgboost import XGBClassifier
        test_xgb = XGBClassifier(device="cuda", n_estimators=1)
        test_xgb.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        print("   🚀 GPU CUDA detectada e habilitada para XGBoost!")
    except Exception:
        print("   💻 Treinamento via CPU otimizado (n_jobs=-1)")

    # Checar diretórios
    dirs = [
        "dataSet/hsi_original",
        "dataSet/data-preProcessed/pre_processing",
        "features_selected",
        "ml/data",
        "ml/models",
        "ml/results",
        "ml/reports",
        "logs",
        "figures",
    ]
    print("\n📁 Estrutura de Diretórios:")
    for d in dirs:
        p = BASE_DIR / d
        status = "✅ Presente" if p.exists() else "⚠️  Não criado"
        print(f"   {status:12s} : {d}")
        
    # Checar dados
    hsi_dir = BASE_DIR / "dataSet" / "hsi_original"
    samples = [f.name for f in hsi_dir.iterdir() if f.is_dir()] if hsi_dir.exists() else []
    print(f"\n🔬 Amostras HSI encontradas em {hsi_dir.name}: {len(samples)}")
    if samples:
        for s in samples[:5]:
            print(f"      - {s}")
        if len(samples) > 5:
            print(f"      ... e mais {len(samples)-5} amostras")
    else:
        print("   💡 Dica: Adicione suas pastas HSI em 'dataSet/hsi_original' ou execute 'python run.py mock' para criar dados de teste.")

    print("=" * 70)
    return all_ok


def init_dirs():
    """Cria todos os diretórios estruturais da pesquisa."""
    print("📁 Inicializando estrutura de diretórios...")
    dirs = [
        "dataSet/hsi_original",
        "dataSet/data-preProcessed/pre_processing",
        "features_selected",
        "ml/data",
        "ml/models",
        "ml/results",
        "ml/reports",
        "logs",
        "figures",
    ]
    for d in dirs:
        p = BASE_DIR / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {d}")
    print("✨ Estrutura de pastas pronta!")


def generate_mock_data():
    """
    Gera um dataset sintético mínimo de imagens HSI (com headers HDR e cubos raw)
    para validação e testes automatizados sem necessitar de arquivos pesados.
    """
    init_dirs()
    print("\n🧪 Gerando amostras sintéticas de teste HSI...")
    import spectral.io.envi as envi
    
    samples = [
        ("CLINIC_351_1", 0),  # Amostra clínica (sensível)
        ("CLINIC_420_1", 0),  # Amostra clínica (sensível)
        ("ATCC13_RES_1", 1),  # Amostra ATCC resistente
        ("ATCC16_RES_1", 1),  # Amostra ATCC resistente
    ]
    
    n_rows, n_cols, n_bands = 80, 80, 100
    wavelengths = [str(round(w, 2)) for w in np.linspace(900, 1700, n_bands)]
    
    for sample_name, label in samples:
        sample_dir = BASE_DIR / "dataSet" / "hsi_original" / sample_name / "capture"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        hdr_path = sample_dir / f"{sample_name}.hdr"
        raw_path = sample_dir / f"{sample_name}.raw"
        
        # Gerar espectros sintéticos com perfil circular central
        data = np.zeros((n_rows, n_cols, n_bands), dtype=np.float32)
        y, x = np.ogrid[:n_rows, :n_cols]
        center_y, center_x = n_rows // 2, n_cols // 2
        radius = 25
        mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        
        base_spectrum = np.linspace(0.2, 0.8, n_bands) + (0.15 if label == 1 else 0.05)
        for r in range(n_rows):
            for c in range(n_cols):
                if mask[r, c]:
                    noise = np.random.normal(0, 0.02, n_bands)
                    data[r, c, :] = np.clip(base_spectrum + noise, 0.01, 1.0)
                else:
                    data[r, c, :] = np.random.normal(0.02, 0.005, n_bands)
        
        # Salvar arquivo ENVI (HDR + RAW)
        metadata = {
            "samples": n_cols,
            "lines": n_rows,
            "bands": n_bands,
            "header offset": 0,
            "file type": "ENVI Standard",
            "data type": 4,  # float32
            "interleave": "bsq",
            "byte order": 0,
            "wavelength": wavelengths,
        }
        
        envi.save_image(str(hdr_path), data, metadata=metadata, ext=".raw", force=True)
        print(f"   ✅ Amostra sintética gerada: {sample_name} (Resistente: {label == 1})")
        
    print("🎉 Amostras sintéticas criadas com sucesso em 'dataSet/hsi_original'!")


def run_script(script_path: Path, desc: str):
    """Executa um script Python do pipeline."""
    print("\n" + "=" * 70)
    print(f"🚀 EXECUTANDO: {desc}")
    print(f"   Script: {script_path.relative_to(BASE_DIR)}")
    print("=" * 70 + "\n")
    
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    if res.returncode != 0:
        print(f"\n❌ Erro durante a execução de {desc} (Código: {res.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Gerenciador e Executor do Ambiente de Pesquisa IC-2025",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=[
            "check",
            "init",
            "mock",
            "etl",
            "split",
            "train",
            "evaluate",
            "all",
        ],
        help=(
            "Ação a ser executada:\n"
            "  check     : Diagnóstico do ambiente e dependências\n"
            "  init      : Cria estrutura de pastas\n"
            "  mock      : Cria dataset sintético para testes rápidos\n"
            "  etl       : Pré-processamento (4 branches) e seleção de features (main.py)\n"
            "  split     : Divide datasets em treino/teste por imagem (ml/split_dataset.py)\n"
            "  train     : Treina modelos XGBoost com Grid Search (ml/train.py)\n"
            "  evaluate  : Avalia no conjunto de teste externo por imagem (ml/evaluate.py)\n"
            "  all       : Executa pipeline completo: etl -> split -> train -> evaluate"
        ),
    )
    
    args = parser.parse_args()
    
    if args.action == "check":
        check_env()
    elif args.action == "init":
        init_dirs()
    elif args.action == "mock":
        generate_mock_data()
    elif args.action == "etl":
        init_dirs()
        run_script(BASE_DIR / "main.py", "Pipeline ETL (4 branches) e Seleção de Features")
    elif args.action == "split":
        init_dirs()
        run_script(BASE_DIR / "ml" / "split_dataset.py", "Divisão Treino/Teste por Imagem")
    elif args.action == "train":
        init_dirs()
        run_script(BASE_DIR / "ml" / "train.py", "Treinamento XGBoost com Grid Search")
    elif args.action == "evaluate":
        init_dirs()
        run_script(BASE_DIR / "ml" / "evaluate.py", "Avaliação no conjunto de teste externo")
    elif args.action == "all":
        init_dirs()
        steps = [
            (BASE_DIR / "main.py", "1/4 - Pipeline ETL (4 branches) e Seleção de Features"),
            (BASE_DIR / "ml" / "split_dataset.py", "2/4 - Divisão Treino/Teste por Imagem"),
            (BASE_DIR / "ml" / "train.py", "3/4 - Treinamento de Modelos (Grid Search)"),
            (BASE_DIR / "ml" / "evaluate.py", "4/4 - Avaliação no Teste Externo"),
        ]
        for script, desc in steps:
            success = run_script(script, desc)
            if not success:
                print(f"⚠️  Interrompendo execução no passo: {desc}")
                sys.exit(1)
        print("\n🎉 Pipeline completo finalizado com sucesso!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        check_env()
    else:
        main()
