from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import subprocess
import sys
import logging

# 📌 Diretórios
BASE_DIR = "/mnt/c/Users/jefferson.faria/Documents/GitHub/ic-2025"

# 🔧 Logger do Airflow
logger = logging.getLogger(__name__)


def run_etl_pipeline(**kwargs):
    """Executa o pipeline ETL (pré-processamento + features)"""
    script_path = os.path.join(BASE_DIR, "main.py")
    logger.info(f"🚀 Executando ETL: {script_path}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR  # ✅ Define diretório de trabalho
    )
    
    logger.info(result.stdout)
    
    if result.returncode != 0:
        logger.error(result.stderr)
        raise Exception(f"ETL falhou: {result.stderr}")
    
    return "ETL concluído com sucesso"


def run_split_dataset(**kwargs):
    """Divide dataset em treino/teste"""
    script_path = os.path.join(BASE_DIR, "ml", "split_dataset.py")
    logger.info(f"✂️ Dividindo dataset: {script_path}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )
    
    logger.info(result.stdout)
    
    if result.returncode != 0:
        logger.error(result.stderr)
        raise Exception(f"Split falhou: {result.stderr}")
    
    return "Split concluído"


def run_training(**kwargs):
    """Treina modelo XGBoost com Grid Search"""
    script_path = os.path.join(BASE_DIR, "ml", "train.py")
    logger.info(f"🌲 Treinando modelo: {script_path}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )
    
    logger.info(result.stdout)
    
    if result.returncode != 0:
        logger.error(result.stderr)
        raise Exception(f"Treinamento falhou: {result.stderr}")
    
    return "Treinamento concluído"


def run_evaluation(**kwargs):
    """Avalia modelo e gera métricas"""
    script_path = os.path.join(BASE_DIR, "ml", "evaluate.py")
    logger.info(f"📊 Avaliando modelo: {script_path}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )
    
    logger.info(result.stdout)
    
    if result.returncode != 0:
        logger.error(result.stderr)
        raise Exception(f"Avaliação falhou: {result.stderr}")
    
    return "Avaliação concluída"


# ⚙️ Configurações padrão
default_args = {
    "owner": "jefferson",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# 🪄 DAG
with DAG(
    dag_id="hsi_full_pipeline",
    default_args=default_args,
    description="Pipeline HSI: ETL → Split → Train → Evaluate",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["HSI", "ETL", "XGBoost", "ML"],
) as dag:

    # 📍 Task 1: ETL (pré-processamento + extração de features)
    task_etl = PythonOperator(
        task_id="etl_pipeline",
        python_callable=run_etl_pipeline,
    )

    # 📍 Task 2: Split treino/teste
    task_split = PythonOperator(
        task_id="split_dataset",
        python_callable=run_split_dataset,
    )

    # 📍 Task 3: Treinamento com Grid Search
    task_train = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    # 📍 Task 4: Avaliação e métricas
    task_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation,
    )

    # 🔗 Dependências (sequencial)
    task_etl >> task_split >> task_train >> task_evaluate