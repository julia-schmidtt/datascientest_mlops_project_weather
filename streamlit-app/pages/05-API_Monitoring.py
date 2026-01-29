import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("API Monitoring")

st.markdown("""
This page displays metrics and performance data from the Weather Prediction API which is displayed on a Grafana dashboard.
""")

# Get Grafana URL from .env
GRAFANA_HOST = os.getenv("GRAFANA_HOST", "localhost")
GRAFANA_PORT = os.getenv("GRAFANA_PORT", "3000")
GRAFANA_DASHBOARD_UID = os.getenv("GRAFANA_DASHBOARD_UID", "")

# Complete Grafana URL
GRAFANA_URL = f"http://{GRAFANA_HOST}:{GRAFANA_PORT}"

st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("Grafana Dashboard")

with col2:
    # Time Range
    time_range = st.selectbox(
        "Time Range",
        ["Last 5 minutes", "Last 15 minutes", "Last 30 minutes", "Last 1 hour", "Last 6 hours", "Last 24 hours"],
        index=1  # Default: Last 15 minutes
    )

with col3:
    # Refresh
    auto_refresh = st.checkbox("Refresh every 5s", value=False)

# Time Range Mapping
time_map = {
    "Last 5 minutes": "now-5m",
    "Last 15 minutes": "now-15m",
    "Last 30 minutes": "now-30m",
    "Last 1 hour": "now-1h",
    "Last 6 hours": "now-6h",
    "Last 24 hours": "now-24h"
}

col1, col2, col3= st.columns(3)

with col1:
    st.link_button("Grafana", GRAFANA_URL, use_container_width=True)

with col2:
    st.link_button("Prometheus", f"http://{GRAFANA_HOST}:9090", use_container_width=True)

with col3:
    if st.button("Refresh Page", use_container_width=True):
        st.rerun()

# Check if Dashboard UID is configured
if not GRAFANA_DASHBOARD_UID:
    st.warning("Grafana Dashboard UID is not configured")
    st.info("""
    To display the dashboard:
    1. Find your dashboard UID in Grafana (in the URL between `/d/` and the next `/`)
    2. Add it to your `.env` file:
```
       GRAFANA_DASHBOARD_UID=your-dashboard-uid
```
    3. Restart the Streamlit container
    """)

    st.markdown("### Manual Access")
    st.link_button("Open Grafana", GRAFANA_URL, use_container_width=True)

else:
    from_time = time_map[time_range]
    to_time = "now"
    refresh_param = "&refresh=5s" if auto_refresh else ""

    grafana_embed_url = f"{GRAFANA_URL}/d/{GRAFANA_DASHBOARD_UID}?orgId=1&from={from_time}&to={to_time}&kiosk=tv{refresh_param}"


    # Embed Grafana Dashboard
    st.markdown(f"""
        <iframe
            src="{grafana_embed_url}"
            width="100%"
            height="1000"
            frameborder="0"
            style="border-radius: 0.5rem;">
        </iframe>
    """, unsafe_allow_html=True)

    st.markdown("---")


# Info Box
with st.expander("About the Monitoring"):
    st.markdown("""

    **Weather API Status**
    - UP/DOWN indicator
    - Data from /health endpoint
    
    
    **Average Response Time**
    - Displayed Per endpoint


    **Alerts**
    - Configured with Prometheus
    - APIEndpointHighLatency: triggered when response time exceeds threshold
    - ModelQualityLow: alerts on model performance degradation, F1 score < 0.6

                
    **API Request Rate by Endpoint**
    - API traffic
    - Real-time request frequency

                
    **API Requests by Endpoint**  
    - Cumulative request count

                         
    **Model Metrics**
    - Metrics for current production model
    - Split ID: data split used for training 
    - F1 Score: harmonic mean of precision and recall
    - Accuracy: overall prediction accuracy
    - Precision: ratio of true positive predictions
    - Recall: ratio of correctly identified positives
    - ROC-AUC: area under the ROC curve


    **System Resources**
    - Memory Usage: RAM consumption
    - Disk Usage: storage utilization
    - CPU Usage: per-core CPU utilization


    **Data Sources:**
    - Prometheus scrapes metrics from FastAPI, Node Exporter, and MLflow
    - Grafana visualizes and alerts on collected metrics
    """)
