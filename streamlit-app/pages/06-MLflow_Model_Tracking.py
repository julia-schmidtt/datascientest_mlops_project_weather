import streamlit as st
import requests
import os
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# Load environment variables
load_dotenv()

st.title("MLflow Model Tracking")

st.markdown("""
Track and compare machine learning experiments, model versions, and performance metrics via DagsHub MLflow.
""")

# Get URLs from .env
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

# DagsHub MLflow Configuration
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")

# Extract repo info from tracking URI
if MLFLOW_TRACKING_URI:
    mlflow_base_url = MLFLOW_TRACKING_URI.replace(".mlflow", "")
    dagshub_experiments_url = f"{MLFLOW_TRACKING_URI}/#/experiments"
    dagshub_models_url = f"{MLFLOW_TRACKING_URI}/#/models"
else:
    mlflow_base_url = ""
    dagshub_experiments_url = ""
    dagshub_models_url = ""

# Authentication für DagsHub API
auth = HTTPBasicAuth(DAGSHUB_USERNAME, DAGSHUB_TOKEN) if DAGSHUB_USERNAME and DAGSHUB_TOKEN else None


# Check Configuration
if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN or not MLFLOW_TRACKING_URI:
    st.warning("DagsHub/MLflow is not fully configured")
    st.info("""
    Please add the following to your `.env` file:
```
    DAGSHUB_USERNAME=your-username
    DAGSHUB_TOKEN=your-token
    MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
```
    """)
    st.stop()

tab1, tab2 = st.tabs(["Current Production Model", "MLflow History"])

with tab1:
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            result = response.json()
            
            # Daten sind unter result['model']
            model_info = result.get('model', {})
            
            # Extrahiere Version und andere Infos
            version = model_info.get('version', 'N/A')
            metrics = model_info.get('metrics', {})
            params = model_info.get('params', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Version", version)
            with col2:
                split_id = params.get('split_id', 'N/A')
                st.metric("Split ID", split_id)
            with col3:
                accuracy = metrics.get('accuracy', 0)
                if isinstance(accuracy, (int, float)) and accuracy > 0:
                    st.metric("Accuracy", f"{accuracy*100:.1f}%")
                else:
                    st.metric("Accuracy", "N/A")
            with col4:
                f1 = metrics.get('f1_score', 0)
                if isinstance(f1, (int, float)) and f1 > 0:
                    st.metric("F1 Score", f"{f1*100:.1f}%")
                else:
                    st.metric("F1 Score", "N/A")
            
            st.markdown("---")

            col1, col2 = st.columns(2)
            
            with col1:
                with st.container(border=True):
                    st.markdown("**More Model Metrics**")
                    
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    roc_auc = metrics.get('roc_auc', 0)
                    
                    if isinstance(precision, (int, float)) and precision > 0:
                        st.text(f"Precision: {precision*100:.1f}%")
                    else:
                        st.text("Precision: N/A")
                    
                    if isinstance(recall, (int, float)) and recall > 0:
                        st.text(f"Recall: {recall*100:.1f}%")
                    else:
                        st.text("Recall: N/A")
                    
                    if isinstance(roc_auc, (int, float)) and roc_auc > 0:
                        st.text(f"ROC-AUC: {roc_auc*100:.1f}%")
                    else:
                        st.text("ROC-AUC: N/A")
            
            with col2:
                with st.container(border=True):
                    st.markdown("**Additional Model Information**")
                    
                    model_name = model_info.get('model_name', 'N/A')
                    st.text(f"Algorithm: {model_name}")
                    
                    registered_at = model_info.get('registered_at', 'N/A')
                    if isinstance(registered_at, int):
                        from datetime import datetime
                        try:
                            date_str = datetime.fromtimestamp(registered_at).strftime('%Y-%m-%d')
                            st.text(f"Training Date: {date_str}")
                        except:
                            st.text("Training Date: N/A")
                    else:
                        st.text("Training Date: N/A")
                    
                    run_id = model_info.get('run_id', 'N/A')
                    if run_id != 'N/A' and len(run_id) > 16:
                        st.text(f"MLflow Run ID: {run_id[:16]}...")
                    else:
                        st.text(f"MLflow Run ID: {run_id}")
            
            with st.expander("Full Model Details from Weather Prediction API endpoint `/model/info`"):
                st.json(result)
                
        else:
            st.error(f"Could not fetch model info from API (Status: {response.status_code})")
            st.info("Make sure the API is running and the `/model/info` endpoint is available")
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API")
        st.info(f"Make sure the API is running on {API_URL}")
    except requests.exceptions.Timeout:
        st.error("Request timeout")
    except Exception as e:
        st.error(f"Error: {str(e)}")

with tab2:
    st.info("View Modeling History on DagsHub MLflow UI by clicking on the buttons below.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if dagshub_experiments_url:
            st.link_button("View Experiments on MLflow", dagshub_experiments_url, use_container_width=True)
    
    with col2:
        if dagshub_models_url:
            st.link_button("View Models on MLflow", dagshub_models_url, use_container_width=True)
