from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow import DAG
import requests
import json

my_dag = DAG(
    dag_id='process_next_split_dag',
    description='process the next split ',
    tags=['automation', 'mlops_weather_project'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=0),
    }
)

# definition of the function to execute
def call_next_split_endpoint():
    url = "http://fastapi_app:8000/pipeline/next-split"
    response = requests.post(url)

    http_status = response.status_code
    body = response.text
    if http_status != 200:
        raise Exception(f"Request failed with status code {http_status}: {response.text}")
    
    else:
        print("Request succeeded.")
        try:
            paresed = json.loads(body)
            print(json.dumps(paresed, indent=2))
        except Exception as e:
            print(body)

    return response.text


my_task = PythonOperator(
    task_id='process_next_split_dag',
    python_callable=call_next_split_endpoint,
    dag=my_dag
)