from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import subprocess

# Caminhos
BASE_DIR = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025"
SCRIPT_DIR = f"{BASE_DIR}/etl"  # ajuste conforme onde estão seus scripts
HSI_ROOT = f"{BASE_DIR}/dataSet/hsi_original"
OUTPUT_DIR = f"{BASE_DIR}/dataSet/data-preProcessed/pre_processing"

# Função genérica para chamar scripts Python existentes
def run_script(script_name: str):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    subprocess.run(["python", script_path], check=True)

# Configuração da DAG
default_args = {
    "owner": "jefferson",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="hsi_preprocessing_pipeline",
    default_args=default_args,
    description="Pipeline ETL para imagens hiperespectrais (ROI, MSC, Log, Derivada, Noise e Features)",
    schedule_interval=None,  # ou "0 3 * * *" para rodar todo dia 03:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["HSI", "ETL", "ML"],
) as dag:

    extract_roi_task = PythonOperator(
        task_id="extract_roi",
        python_callable=run_script,
        op_args=["extract_roi.py"],
    )

    msc_task = PythonOperator(
        task_id="msc_signal",
        python_callable=run_script,
        op_args=["msc_signal.py"],
    )

    log_task = PythonOperator(
        task_id="log_transform",
        python_callable=run_script,
        op_args=["log_transform.py"],
    )

    derivative_task = PythonOperator(
        task_id="derivative_transform",
        python_callable=run_script,
        op_args=["derivative_transform.py"],
    )

    noise_task = PythonOperator(
        task_id="noise_band_removal",
        python_callable=run_script,
        op_args=["noise_band_removal.py"],
    )

    features_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_script,
        op_args=["feature_engineering.py"],
    )

    band_selection_task = PythonOperator(
        task_id="band_selection",
        python_callable=run_script,
        op_args=["band_selection.py"],
    )

    # Dependências (ordem de execução)
    extract_roi_task >> msc_task >> log_task >> derivative_task >> noise_task >> features_task >> band_selection_task
