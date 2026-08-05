from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default settings applied to all tasks in this DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
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
    # We mapped the scripts folder to /opt/airflow/scripts in the docker-compose.yml
    run_finetuning = BashOperator(
        task_id='execute_pytorch_training',
        bash_command='python /opt/airflow/scripts/train_combiner.py',
    )

    # If we had multiple steps (like Data Extraction -> Training -> Deployment), 
    # we would chain them here using >> (e.g., extract_data >> run_finetuning >> deploy_model)
    run_finetuning