from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'combiner_finetuning_pipeline',
    default_args=default_args,
    description='Automated pipeline to fine-tune the Combiner model on user clicks',
    schedule_interval='@daily', # Automatically runs every midnight
    start_date=datetime(2026, 8, 4),
    catchup=False,
    tags=['machine_learning', 'fashion_iq'],
) as dag:

    # Task 1: Execute the PyTorch training script
    run_finetuning = BashOperator(
        task_id='execute_pytorch_training',
        bash_command='python /opt/airflow/scripts/train_combiner.py',
    )

    # Task 2: Ping FastAPI to download and load the new weights
    trigger_fastapi_reload = BashOperator(
        task_id='trigger_fastapi_reload',
        bash_command='curl -X POST http://api:8000/reload-model',
    )

    run_finetuning