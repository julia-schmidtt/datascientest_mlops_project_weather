import streamlit as st

st.title("Project Architecture")

st.markdown("""
This page explains the Docker-based architecture and data flow of the MLOps weather prediction system.
""")

tab1, tab2, tab3 = st.tabs(["Overview", "Architecure Diagram", "Data Evolution & Monitoring"])


with tab1:
    st.markdown("""
The whole system runs in Docker containers orchestrated by Docker Compose. 
Here, the present Docker containers are described. The FastAPI container has a special startup sequence.
""")

    containers = [
        {
            "name": "FastAPI",
            "port": "8000",
            "description": """
- Core API for predictions, model training, and pipeline usage
- Container startup steps:
    1. MLflow cleanup: all existing models are archived to ensure a fresh start for simulating data evolution and model lifecycle
    2. Downloads raw weather dataset from Kaggle
    3. Raw data preprocessing with the identified steps: saves processed data, scaler and default values
    4. Creation of year-based training data splits for manual training
    5. API start
"""
        },
        {
            "name": "Node Exporter",
            "port": "9100",
            "description": "- Exports host system metrics (CPU, memory, disk usage)"
        },
        {
            "name": "Prometheus",
            "port": "9090",
            "description": "- Scrapes and stores metrics from API and system"
        },
        {
            "name": "Grafana",
            "port": "3000",
            "description": "- Visualizes API performance, system resources, and alerts"
        },
        {
            "name": "Airflow",
            "port": "8080 (optional)",
            "description": "- Schedules automated pipeline tasks (resource-intensive). Use cron if resources are limited."
        },
        {
            "name": "Streamlit",
            "port": "8501",
            "description": "- Project presentation and user interface for interacting with the API"
        }
    ]

    for container in containers:
        with st.container(border=True):
            st.markdown(f"**{container['name']}**")
            st.markdown(f"`Port {container['port']}`")
            st.markdown(container["description"])
            

with tab2:
    st.image("/app/images/architecture.png", use_column_width=True)    

with tab3:
    
    st.markdown("""
    Data evolution over time:
    - Simulated by periodically calling pipeline API endpoints to process new data splits
    - Possible to perform with Airflow, if enough resources are present (Option 1)
    - If resources are limited: define cron job (Option 2)
    - Procedure should simulate real-world scenarios where new data arrives in certain time intervals

    API pipeline endpoint without data drift detection:
    - Model training for every new data arrival
    - Performance comparison of new model with  production model performance
    - If performance of new model is better, it becomes production model and the old production model is archived
    - If performance of new model is worse, it is archived and the production model does not change
                
    API pipeline endpoint with data drift detection:
    - Conditional model training: comparison of new data with training data of current production model
    - If data drift is detected, the model is retrained with the new data (after training: model performance comparison with production model, decision like in endpoint without drift detection)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("**Option 1: Airflow (resource-hungry)**")
            st.markdown("""
            - Schedules API pipeline endpoints via DAGs
            - Use only when sufficient resources are available
            """)
    
    with col2:
        with st.container(border=True):
            st.markdown("**Option 2: Cron Jobs**")
            st.markdown("""
            - Simple scheduled curl commands for one of the two API pipeline endpoints in defined intervals
            - Use when only limited resources are available
            - **Example**:
                ```bash
                # Every hour
                0 * * * * curl -X POST http://localhost:8000/pipeline/next-split-drift-detection
                ```
            """)
    
