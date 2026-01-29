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
    st.markdown("""INSERT DIAGRAM""")
    

with tab3:
    
    st.subheader("Simulating Data Evolution and Triggering (Conditional) Model Training")
    
    st.markdown("""
    Data evolution over time is simulated by periodically calling pipeline API endpoints to process new data splits.
    This can be performed with Airflow, if enooigh resources are present (Option 1).
    If the resources are limited cron jobs can be defined (Option 2). 
    This procedure should simulate real-world scenarios where new data arrives in certain intervals.
                
    There are two API pipeline endpoints present, one without data drift detection and another with data drift detection before training (conditional training).
    If the endpoint without data drift detection is used, the model is trained every time after new data arrived.
    The performance of the new model is compared to the production model performance. 
    If the performance of the new model is better, it becomes production model.
    If the performance is worse, it is archived and the production model does not change.
                
    If the endpoint with data drift detection is used, the newly arrived data is compared to the data of the current production model.
    If data drift is detected, the model is retrained with the new data. Afterwards the performance of the new model is compared to the production model performance. 
    If the performance of the new model is better, it becomes the new production model. 
    If the performance is worse, the new model is archived.
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
    
