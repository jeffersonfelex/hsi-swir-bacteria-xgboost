from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os
import shutil
import subprocess

INPUT_DIR = '/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_raw'   # <-- ajuste conforme seu sistema
OUTPUT_DIR = '/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/ETL/data_processed'
SCRIPTS_DIR = '/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025/scripts'

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
    'catchup': False
}

def get_first_file():
    """Retorna o primeiro arquivo .csv encontrado no diretório de entrada"""
    for f in os.listdir(INPUT_DIR):
        if f.endswith('.csv'):
            return f
    return None

def run_all_scripts():
    file = get_first_file()
    if not file:
        raise ValueError("Nenhum arquivo encontrado na pasta de entrada.")

    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, f'processed_{file}')

    for script in sorted(os.listdir(SCRIPTS_DIR)):
        if script.endswith('.py'):
            script_path = os.path.join(SCRIPTS_DIR, script)
            cmd = ['python3', script_path, input_path, output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f'Erro no script {script}: {result.stderr}')

    # Move o arquivo original para evitar reprocessamento
    shutil.move(input_path, os.path.join(INPUT_DIR, f'processed_{file}'))

with DAG('etl_pipeline_with_file_sensor', default_args=default_args, schedule_interval='@daily') as dag:

    wait_for_file = FileSensor(
        task_id='wait_for_csv_file',
        filepath='input',  # O sensor detecta qualquer coisa no diretório
        fs_conn_id='fs_default',  # Configure esta conexão no Airflow Connections
        poke_interval=30,
        timeout=60 * 5,  # 5 minutos
        mode='poke'
    )

    process_scripts = PythonOperator(
        task_id='run_scripts',
        python_callable=run_all_scripts
    )

    wait_for_file >> process_scripts
