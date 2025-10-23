from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import subprocess
import sys

# 📌 Diretórios
BASE_DIR = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025"
SCRIPT_PATH = os.path.join(BASE_DIR, "run_pipeline.py")

# 🛠️ Função para rodar o pipeline completo
def run_full_pipeline(**kwargs):
    print(f"🚀 Iniciando execução do pipeline: {SCRIPT_PATH}")
    result = subprocess.run([sys.executable, SCRIPT_PATH], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Erro na execução do pipeline:")
        print(result.stderr)
        raise Exception(f"Pipeline falhou: {result.stderr}")
    else:
        print("✅ Pipeline executado com sucesso!")
        print(result.stdout)

# ⚙️ Configurações padrão da DAG
default_args = {
    "owner": "jefferson",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

# 🪄 DAG
with DAG(
    dag_id="hsi_full_pipeline",
    default_args=default_args,
    description="Pipeline completo HSI: ETL + Feature Engineering + Seleção de Bandas",
    schedule_interval=None,  # ⏳ Defina cron se quiser agendar
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["HSI", "ETL", "Band Selection"],
) as dag:

    pipeline_task = PythonOperator(
        task_id="run_hsi_pipeline",
        python_callable=run_full_pipeline,
        provide_context=True
    )

    pipeline_task
