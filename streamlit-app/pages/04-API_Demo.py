import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("API Demo")

# Get API URL from .env
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"


# API Status Check
with st.container(border=True):
    st.markdown("**API Status**")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.markdown("The Weather Prediction API is running on Port 8000. Explore the available endpoints below.")
            st.success("API is running")
        else:
            st.error("Weather Prediction API is not responding correctly.")
    except Exception as e:
        st.error(f"API is not reachable: {str(e)}")
        st.info("Make sure the Weather Prediction API is running on port 8000 and your IP address is noted in .env (see README).")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["API Endpoints", "Live Weather Prediction", "README"])

with tab1:
    with st.container(border=True):
        st.markdown("**POST/login**")
        st.code("http://localhost:8000/login", language="text")
        st.markdown("First step: Use login endpoint to get token for API security.")

    st.markdown("---")
 

    st.markdown("""
        <style>
        /* all boxes min height */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 250px;
        }
        </style>
        """, unsafe_allow_html=True)

    endpoints_with_button = [
        ("GET", "/", "Returns API information and available endpoints."),
        ("GET", "/health", "Health check endpoint."),
        ("GET", "/model/info", "Returns information regarding current production model."),
        ("GET", "/metrics", "Returns API metrics."),
    ]

    endpoints_without_button = [
        ("POST", "/model/refresh", "Reloads current production model."),
        ("POST", "/train", "Manually train model for specific training data split."),
        ("POST", "/predict/simple", "Prediction for 'Rain Tomorrow' of current production model, 5 inputs required (location, date, min_temp, max_temp, rain_today)."),
        ("POST", "/predict", "Prediction of 'Rain Tomorrow' of current production model, 110 inputs required."),
        ("POST", "/pipeline/next-split", "Automated pipeline for triggering data growth, model training and new decision regarding production model."),
        ("POST", "/pipeline/next-split-drift-detection", "Automated pipeline for triggering data growth, data drift detection, conditional training if data drift is detected and decision regarding production model. If no data drift is present, no training will be performed and the current production model will not change."),
    ]




    for i in range(0, len(endpoints_with_button), 2):
        col1, col2 = st.columns(2)

        with col1:
            method, path, description = endpoints_with_button[i]

            with st.container(border=True):
                st.markdown(f"**{method} {path}**")
                st.code(f"http://localhost:8000{path}", language="text")
                st.markdown(description)

                button_clicked = st.button("Try it", key=f"btn_{path}")

            if button_clicked:
                try:
                    response = requests.get(f"{API_URL}{path}", timeout=5)

                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '')

                        if 'application/json' in content_type:
                            st.json(response.json())
                        else:
                            st.code(response.text, language="text")
                    else:
                        st.error(f"Error {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(str(e))

        if i + 1 < len(endpoints_with_button):
            with col2:
                method, path, description = endpoints_with_button[i + 1]

                with st.container(border=True):
                    st.markdown(f"**{method} {path}**")
                    st.code(f"http://localhost:8000{path}", language="text")
                    st.markdown(description)

                    button_clicked = st.button("Try it", key=f"btn_{path}")

                if button_clicked:
                    try:
                        response = requests.get(f"{API_URL}{path}", timeout=5)

                        if response.status_code == 200:
                            content_type = response.headers.get('Content-Type', '')

                            if 'application/json' in content_type:
                                st.json(response.json())
                            else:
                                st.code(response.text, language="text")
                        else:
                            st.error(f"Error {response.status_code}")
                            st.code(response.text)
                    except Exception as e:
                        st.error(str(e))

    st.markdown("---")

    for i in range(0, len(endpoints_without_button), 2):
        col1, col2 = st.columns(2)

        with col1:
            method, path, description = endpoints_without_button[i]

            with st.container(border=True):
                st.markdown(f"**{method} {path}**")
                st.code(f"http://localhost:8000{path}", language="text")
                st.markdown(description)

        if i + 1 < len(endpoints_without_button):
            with col2:
                method, path, description = endpoints_without_button[i + 1]

                with st.container(border=True):
                    st.markdown(f"**{method} {path}**")
                    st.code(f"http://localhost:8000{path}", language="text")
                    st.markdown(description)

with tab2:

    st.markdown("Try out the weather prediction API for locations in Australia. Here the predict/simple endpoint is demonstrated, therefore only 5 inputs are required.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            location = st.selectbox(
                "Location",
                ["Sydney", "Berlin", "Melbourne", "Brisbane", "Perth", "Adelaide", "Hobart", "Canberra", "Darwin"]
            )
            date = st.date_input("Date", value=None)

        with col2:
            min_temp = st.number_input("Min Temperature (°C)", value=10.0, step=0.1)
            max_temp = st.number_input("Max Temperature (°C)", value=20.0, step=0.1)

        with col3:
            rain_today = st.selectbox("Rain Today", ["No", "Yes"])

        submit = st.form_submit_button("🌦️ Predict Rain Tomorrow", use_container_width=True)

        if submit:
            #  Validation
            if date is None:
                st.error("Please select a date")
            elif min_temp >= max_temp:
                st.error("Min temperature must be less than max temperature")
            else:
                # Mapping: "Yes" -> 1, "No" -> 0
                rain_today_value = 1 if rain_today == "Yes" else 0

                payload = {
                    "location": location,
                    "date": str(date),
                    "min_temp": min_temp,
                    "max_temp": max_temp,
                    "rain_today": rain_today_value
                }

                try:
                    with st.spinner("Making prediction..."):
                        response = requests.post(
                            f"{API_URL}/predict/simple",
                            json=payload,
                            timeout=10
                        )

                    if response.status_code == 200:
                        result = response.json()

                        st.success("Prediction successful!")

                        # Mapping
                        prediction_value = result.get("prediction", 0)
                        prediction = "Yes" if prediction_value == 1 else "No"

                        # Show results
                        col1, col2 = st.columns(2)

                        with col1:
                            with st.container(border=True):
                                if prediction == "Yes":
                                    st.markdown("### 🌧️  Rain Expected Tomorrow")
                                else:
                                    st.markdown("### ☀️ No Rain Expected Tomorrow")

                        with col2:
                            with st.container(border=True):
                                prob_rain = result.get('probability_rain', 0) * 100
                                st.metric(
                                    label='Model Prediction',
                                    value=prediction
                                )
                                st.metric(
                                    label="Rain Probability for Tomorrow",
                                    value=f"{prob_rain:.1f}%"
                                )

                        # Request/Response Details
                        with st.expander("View Request & Response Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Request:**")
                                st.json(payload)
                            with col2:
                                st.markdown("**Response:**")
                                st.json(result)

                    else:
                        st.error(f"Error: HTTP {response.status_code}")
                        st.code(response.text)

                except requests.exceptions.Timeout:
                    st.error("Request timeout - API took too long to respond")
                except requests.exceptions.ConnectionError:
                    st.error("Connection error - Cannot reach API")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab3:
    st.markdown("""
# Rain Prediction in Australia - MLOps Project

This project implements a MLOps pipeline for rain prediction in Australia.

---

## Table of Contents

- [Project Organization](#project-organization)
- [Setup](#setup)
  - [Option A: Docker Container Setup](#option-a-docker-container-setup)
  - [Option B: Local Setup](#option-b-local-setup)
- [API Usage](#api-usage)
  - [Option A: Docker Container API Usage](#option-a-docker-container-api-usage)
  - [Option B: Local API Usage](#option-b-local-api-usage)
- [Additional Information](#additional-information)
  - [1. XGBoost Hyperparameters](#1-xgboost-hyperparameters-in-paramsyaml)
  - [2. SMOTE Class Balancing](#2-smote-class-balancing)
  - [3. Data Drift Monitoring](#3-data-drift-monitoring-paramsyaml)
  - [4. MLflow Integration](#4-mlflow-integration)
  - [5. Experiment Organization](#5-experiment-organization-on-mlflow)

---
---

## Project Organization

```
├── dags/
│   ├── process_next_split_dag.py                     # DAG for API endpoint pipeline/next-split for Airflow
│   └── process_next_split_with_driftdetection_dag.py # DAG for API endpoint pipeline/next-split-drift-detection for Airflow
│
├── data/
│   ├── raw/                                          # Original raw data (weatherAUS.csv)
│   ├── interim/                                      # Preprocessed data
│   ├── processed/                                    # Train/test splits, filled missing values, One-Hot encoded, scaled, ready for modeling
│   ├── training_data_splits_by_year/                 # Temporal splits (manual creation) of training data (2008, 2008+2009, 2008+2009+2010...) used by API training endpoint
│   └── automated_splits/                             # Automated temporal splits of training data, created and used by API pipeline endpoints
│
├── deployment/
│   ├── grafana/
│   │   ├── dashboards/                               # json for Grafana dashboard which is automatically loaded
│   │   └── provisioning/                             # Files for automatic loading of Grafana dashboard
│   └── prometheus/
│       ├── prometheus.yaml                           # Prometheus implementation
│       └── rules                                     # Prometheus alert rules
│
├── models/
│   ├── scaler.pkl                                    # StandardScaler extracted from preprocessing phase for processing API input for prediction endpoints
│   └── xgboost_model_split_XX.pkl                    # Trained XGBoost models
│
├── monitoring/
│   └── reports/                                      # Evidently data drift detection HTML reports
│
├── src/
│   ├── api/
│   │   ├── main.py                                   # FastAPI application
│   │   ├── defaults.json                             # Training defaults for filling missing API inputs in simplied prediction endpoint, extracted from preprocessing phase
│   │   └── validation_data.json                      # Possible locations and seasons for checking API input, extracted from preprocessing phase
│   ├── config/
│   │   └── __init__.py                               # Parameter management
│   ├── data/
│   │   ├── preprocess.py                             # Full data preprocessing: generates data in data/interim, processed and src/api/defaults.json, validation_data.json
│   │   ├── training_data_splits_by_year.py           # Splits processed training data based on year (manual), generates data in data/training_data_splits_by_year
│   │   └── automation_create_split.py                # Automated incremental training data split creation used by pipeline API endpoints
│   ├── monitoring/
│   │   └── data_drift.py                             # Data drift detection (Evidently)
│   └── models/
│       ├── train_model.py                            # Model training with MLflow
│       └── predict_model.py                          # Prediction script
│
├── scripts/
│   ├── process_next_split.sh                         # Cron job wrapper script
│   ├── process_next_split_with_drift_detection.sh    # Cron job wrapper script (with drift detection)
│   └── archive_all_models.py                         # Script to archive all models in registry to start without production model
│
├── streamlit-app/
│   ├── images/                                       # Images for Streamlit App
│   ├── pages/                                        # Pages for Streamlit App
│   │   ├── page_1.py
│   │   └── page_2.py
│   ├── tables/                                       # Tables for Streamlit App
│   ├── requirements.txt                              # Requirements for Streamlit App
│   └── streamlit-app.py                              # Streamlit App Code
│
├── logs/                                             # Pipeline execution logs
│
├── tests/
│   └── test_api_prediction.py                        # Script for full prediction API endpoint with 110 input features
│
├── params.yaml                                       # Configuration parameters
├── requirements.txt                                  # Python dependencies
├── Dockerfile.api                                    # Dockerfile for API
├── Dockerfile.streamlit                              # Dockerfile for Streamlit App
├── docker-compose.yaml                               # docker-compose file
├── dvc.lock                                          # dvc pipeline
├── dvc.yaml                                          # Description of dvc pipeline
└── README.md
```

---
---

## Setup

>Choose between Docker Container or Local setup based on your needs.

### Option A: Docker Container Setup

#### **Step 1: Clone Repository**

```bash
git clone https://github.com/julia-schmidtt/datascientest_mlops_project_weather.git

cd datascientest_mlops_project_weather
```

#### **Step 2: Configure Environment Variables**

Create `.env` file in the root of your project:

```bash
nano .env
```

Add the following credentials:

```env
# Kaggle credentials (raw dataset will be loaded from Kaggle)
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-token

# DagsHub credentials
DAGSHUB_USERNAME=julia-schmidtt
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/julia-schmidtt/datascientest_mlops_project_weather.mlflow

# Airflow configuration
AIRFLOW_UID=1000
AIRFLOW_GID=0

#IP address and API port (8000) used by Streamlit app
API_HOST=your-IP-address
API_PORT=8000

# IP adress, Grafana port (3000) and Grafana dashboard UID (can be found when opening the dashboard on Grafana in the URL: http://localhost:3000/d/ab1cdef/mlops-project3a-weather-prediction-dashboard?orgId=1&from=now-15m&to=now&timezone=browser&refresh=auto
# In the example URL the UID is `ab1cdef`
GRAFANA_HOST=your-IP-address
GRAFANA_PORT=3000
GRAFANA_DASHBOARD_UID=dashboard-uid
```

> **Get your Kaggle token:** Kaggle → Settings → API Tokens → Generate New Token

> **Get your DagsHub token:** DagsHub → Settings → Tokens


#### **Step 3: Start Docker Containers**

> **IMPORTANT:** Airflow is resource-intensive and will crash normal VMs. Use the VM from the Airflow module for this setup.

Build and start all containers:

```bash
docker compose up -d --build
```

Only build certain containers:
```bash
docker compose up -d service-name-from-docker-compose-file
```

E.g. if you want to build all containers except Airflow containers:
```bash
docker compose up -d fastapi streamlit postgres redis prometheus grafana node-exporter

```
The following services will be available if you start all containers:

| Service | Port | URL | Credentials |
|---------|------|-----|-------------|
| FastAPI | 8000 | http://localhost:8000 | - |
| Airflow | 8080 | http://localhost:8080 | `airflow` / `airflow` |
| Prometheus | 9090 | http://localhost:9090 | - |
| Node Exporter | 9100 | http://localhost:9100 | - |
| Grafana | 3000 | http://localhost:3000 | `admin` / `admin` |
| Streamlit | 8501 | http://localhost:8501 | - |

> **Port Forwarding:** When working on a VM with VSCode or a ssh connection, you may need to forward ports to your local machine.

**Airflow Configuration:**

Ensure proper permissions for Airflow directories:

```bash
sudo chmod -R 777 logs/
sudo chmod -R 777 dags/
sudo chmod -R 777 plugins/
```

You should see the DAGs `process_next_split_dag` and `process_next_split_with_driftdetection_dag` ready for execution at http://localhost:8080.

**Grafana Configuration:**

If the dashboard doesn't load automatically:
1. Navigate to http://localhost:3000
2. Click **Dashboards** → **New** → **Import**
3. Upload the JSON file from `deployment/grafana/dashboards/`

> **Note:** Metrics may take time to appear. Try adjusting the time period if data doesn't show immediately.

#### **Step 4: Stop Docker Containers**

When finished and you want do remove all containers:
```bash
docker compose down
```

Stopping certain containers:
```bash
docker compose stop service-name-from-docker-compose-file
```

Remove certain containers:
```bash
docker compose down service-name-from-docker-compose-file
```

```docker compose stop``` will only stop containers and data inside containers will be kept. If you want to start the containers again use ```docker compose start```.
```docker compose down``` will remove containers and data inside containers will be deleted. If you want to build the containers again use ```docker compose up```.

---

### Option B: Local Setup

#### **Step 1: Clone Repository**

```bash
git clone https://github.com/julia-schmidtt/datascientest_mlops_project_weather.git

cd datascientest_mlops_project_weather
```

#### **Step 2: Configure Environment Variables**

Create `.env` file in the root of the project:

```bash
nano .env
```

Add DagsHub credentials:

```env
DAGSHUB_USERNAME=julia-schmidtt
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/julia-schmidtt/datascientest_mlops_project_weather.mlflow
```

> **Get your DagsHub token:** DagsHub → Settings → Tokens

#### **Step 3: Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **Step 4: Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **Step 5: Initialize DVC**

```bash
dvc init
dvc remote add origin https://dagshub.com/your-username/datascientest_mlops_project_weather.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user your-username
dvc remote modify origin --local password your-token
```

#### **Step 6: Generate Data**

Pull raw data and generate all preprocessing outputs locally:

```bash
# Pull only raw data
dvc pull data/raw/weatherAUS.csv.dvc

# Generate all preprocessing outputs
python src/data/preprocess.py

# Generate manual training splits
python src/data/training_data_splits_by_year.py
```

**This creates:**
- `data/interim/df_preprocessed.csv` - Processed data
- `data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv` - Data ready for modeling
- `models/scaler.pkl` - Required for API predictions
- `src/api/defaults.json` - Default values for missing features
- `src/api/validation_data.json` - Valid locations and seasons
- `data/training_data_splits_by_year/` - 9 cumulative training splits by year

---
---

## API Usage

### Option A: Docker Container API Usage

View API logs:

```bash
docker compose logs -f fastapi
```

---

#### **Available API Endpoints**

##### **1. Health Check**

```bash
curl http://localhost:8000/health
```

##### **2. Model Info**

```bash
curl http://localhost:8000/model/info
```

##### **3. Reload Model**

```bash
curl -X POST http://localhost:8000/model/refresh
```

##### **4. Training Endpoint**

Train model on specific split (example: split 1):

```bash
curl -X POST "http://localhost:8000/train?split_id=1"
```

**MLflow experiment:** `WeatherAUS_YearBased_Training`

##### **5. Simplified Prediction**

Requires only 5 input features:

```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Sydney",
    "date": "2025-01-15",
    "min_temp": 18.0,
    "max_temp": 28.0,
    "rain_today": 0
  }'
```

##### **6. Full Prediction**

Requires 110 input features:

> **Execute inside API Docker container**

```bash
# Enter container
docker exec -it fastapi_app bash

# Run prediction test
python tests/test_api_prediction.py

# Exit container
exit
```

##### **7. Automated Pipeline**

```bash
curl -X POST http://localhost:8000/pipeline/next-split
```

**Complete workflow:**
1. Create next temporal split
2. Train XGBoost model on new split
3. Log to MLflow (metrics, parameters, artifacts)
4. Compare F1 score with production model
5. Auto-promote if better
6. Reload production model in API

**MLflow experiment:** `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`

##### **8. Automated Pipeline with Drift Detection**

```bash
curl -X POST http://localhost:8000/pipeline/next-split-drift-detection
```

**Complete workflow:**
1. Create next temporal split
2. **Data Drift Check:** Compare new split with production model's training data
3. **Conditional Training:**
   - If drift > threshold: Train new model, compare, promote if better
   - If drift < threshold: Skip training to save resources
4. Log to MLflow (metrics, parameters, drift reports)
5. Reload production model in API

**MLflow experiment:** `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`

##### **9. Metrics**
```bash
curl http://localhost:8000/metrics
```

---

#### **Cron Job Automation**

Setup for continuous training of all data splits.

##### **Step 1: Restart API Container**

```bash
docker compose down

docker compose up -d --build
```

##### **Step 2: Configure Cron Job**

> **IMPORTANT:** Use absolute paths! Replace with your actual project directory.

Find your absolute path:

```bash
cd /path/to/your/project

pwd  # Copy this output
```

Edit crontab:

```bash
crontab -e
```

**Option a: Pipeline WITHOUT Drift Detection** (trains every split)

```bash
# Example: every 2 minutes
*/2 * * * * /path/to/scripts/process_next_split.sh >> /path/to/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M).log 2>&1
```

Execution timeline:
```
Time 00:00: Split 1 → Model v20 → Production
Time 00:02: Split 2 → Model v21 → Compare → Promote if better
Time 00:04: Split 3 → Model v22 → Compare → Promote if better
...
Time XX:XX: Split 9 → Model v28 → Compare → Promote if better
```

**Option b: Pipeline WITH Drift Detection** (conditional training)

```bash
# Example: every 2 minutes
*/2 * * * * /path/to/scripts/process_next_split_with_drift_detection.sh >> /path/to/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M).log 2>&1
```

Execution timeline:
```
Time 00:00: Split 1 → Model v20 → Production
Time 00:02: Split 2 → Drift Check → If drift > threshold → Model v21 → Compare → Promote if better
...
```

##### **Step 3: Monitor Execution**

```bash
# List cron jobs
crontab -l

# View logs (latest first)
ls -lth logs/
```

**Log file contents include:**
- Split creation info
- Model metrics (F1, accuracy, precision, recall, ROC-AUC)
- Promotion decision
- Production model version

---

#### **API Grafana Dashboard**

Find API health dashboard at: `http://localhost:3000`

credentials: admin/admin

- API health status
- Average Response Time per Endpoint
- Prometheus Alerts (Prometheus: `http://localhost:8080`)
- API Request Rate by Endpoint
- Total API Requests by Endpoint
- Model Metrics of production model
- Memory usage
- Disk usage
- CPU usage

---

#### **Airflow**

Find Airflow at: `http://localhost:8080`

credentials: airflow/airflow

- important DAGs: ```process_next_split_dag``` and ```process_next_split_with_driftdetection_dag``
- activate and execute DAGs on Airflow UI
- schedule DAGs by implementing it in `dags/process_next_split_dag.py, process_next_split_with_driftdetection_dag.py` (make sure DAG is activated on Airflow UI):

```python
# Change in dags/process_next_split_dag.py
my_dag = DAG(
    dag_id='process_next_split_dag',
    description='process the next split ',
    tags=['automation', 'mlops_weather_project'],
    schedule_interval='*/2 * * * *', # every 2 minutes
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=0),
    }
)

# Change in dags/process_next_split_with_driftdetection_dag.py
my_dag = DAG(
    dag_id='process_next_split_with_driftdetection_dag',
    description='process the next split and perform drift detection',
    tags=['automation', 'mlops_weather_project'],
    schedule_interval='*/2 * * * *', # every 2 minutes
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=0),
    }
)

```

---

#### **Streamlit**

Find Streamlit Application at: `http://localhost:8501`

---

### Option B: Local API Usage

#### **Step 1: Start API**

```bash
python src/api/main.py
```

> API runs on `http://localhost:8000`

#### **Step 2: Use API Endpoints**

##### **1. Health Check**

```bash
curl http://localhost:8000/health
```

##### **2. Model Info**

```bash
curl http://localhost:8000/model/info
```

##### **3. Reload Model**

```bash
curl -X POST http://localhost:8000/model/refresh
```

##### **4. Training Endpoint**

Train model on specific split (example: split 1):

```bash
curl -X POST "http://localhost:8000/train?split_id=1"
```

**MLflow experiment:** `WeatherAUS_YearBased_Training`


##### **5. Simplified Prediction**

Requires only 5 input features:

```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Sydney",
    "date": "2025-01-15",
    "min_temp": 18.0,
    "max_temp": 28.0,
    "rain_today": 0
  }'
```

##### **6. Full Prediction**

Requires 110 input features:

```bash
python tests/test_api_prediction.py
```

##### **7. Automated Pipeline**

```bash
curl -X POST http://localhost:8000/pipeline/next-split
```

**Complete workflow:**
1. Create next temporal split
2. Track with DVC (`dvc add`, `git commit`, `git push`, `dvc push`)
3. Train XGBoost model on new split
4. Log to MLflow (metrics, parameters, artifacts)
5. Compare F1 score with production model
6. Auto-promote if better
7. Reload production model in API

**Output directory:** `data/automated_splits/`
**MLflow experiment:** `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`

##### **8. Automated Pipeline with Drift Detection**

```bash
curl -X POST http://localhost:8000/pipeline/next-split-drift-detection
```

**Complete workflow:**
1. Create next temporal split
2. Track with DVC (`dvc add`, `git commit`, `git push`, `dvc push`)
3. **Data Drift Check:** Compare new split with production model's training data
4. **Conditional Training:**
   - If drift > threshold: Train new model, compare, promote if better
   - If drift < threshold: Skip training to save resources
5. Log to MLflow (metrics, parameters, drift reports)
6. Reload production model in API

**Drift Configuration** (`params.yaml`):

```yaml
monitoring:
  drift_check_enabled: true
  drift_threshold: 0.10  # Adjust threshold as needed
```

**Drift Reports:** `monitoring/reports/`
**Output directory:** `data/automated_splits/`
**MLflow experiment:** `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`

---

#### **Cron Job Automation**

##### **Step 1: Prepare Environment**

Archive all existing models and restart API:

```bash
python scripts/archive_all_models.py
python src/api/main.py
```

##### **Step 2: Clean Automated Splits**

Start fresh with split 1:

```bash
rm -rf data/automated_splits/split_*
rm -f data/automated_splits/metadata.yaml
ls -la data/automated_splits/
```

> **Expected result:** Only `.gitignore` should remain in `data/automated_splits/`

##### **Step 3: Configure Cron Job**

> **IMPORTANT:** Use absolute paths! Replace with your actual project directory.

Find your absolute path:

```bash
cd /path/to/your/project
pwd  # Copy this output
```

Edit crontab:

```bash
crontab -e
```

**Option a: Pipeline WITHOUT Drift Detection** (trains every split)

```bash
# Example: every 2 minutes
*/2 * * * * /path/to/scripts/process_next_split.sh >> /path/to/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M).log 2>&1
```

Execution timeline:
```
Time 00:00: Split 1 → Model v20 → Production
Time 00:02: Split 2 → Model v21 → Compare → Promote if better
Time 00:04: Split 3 → Model v22 → Compare → Promote if better
...
Time XX:XX: Split 9 → Model v28 → Compare → Promote if better
```

**Option b: Pipeline WITH Drift Detection** (conditional training)

```bash
# Example: every 2 minutes
*/2 * * * * /path/to/scripts/process_next_split_with_drift_detection.sh >> /path/to/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M).log 2>&1
```

Execution timeline:
```
Time 00:00: Split 1 → Model v20 → Production
Time 00:02: Split 2 → Drift Check → If drift > threshold → Model v21 → Compare → Promote if better
...
```

##### **Step 4: Monitor Execution**

```bash
# List cron jobs
crontab -l

# View logs (latest first)
ls -lth logs/
```

**Log file contents include:**
- Split creation info
- DVC/Git tracking status
- Model metrics (F1, accuracy, precision, recall, ROC-AUC)
- Promotion decision
- Production model version

---
---

## Additional Information

### 1. XGBoost Hyperparameters in `params.yaml`

```yaml
max_depth: 9
learning_rate: 0.05
n_estimators: 200
colsample_bytree: 0.8
subsample: 0.7
gamma: 0.1
min_child_weight: 3
```

---

### 2. SMOTE Class Balancing

**Original target distribution:**
- 78% No Rain
- 22% Rain

**After SMOTE:**
- 50% No Rain
- 50% Rain

---

### 3. Data Drift Monitoring (`params.yaml`)

- Uses **Evidently** for data drift detection
- Compares new training split against production model's training data
- Training triggered only if drift percentage exceeds threshold

**Configuration:**

```yaml
monitoring:
  drift_check_enabled: true
  drift_threshold: 0.10
```

---

### 4. MLflow Integration

All experiments are tracked on **DagsHub MLflow**:

```
https://dagshub.com/julia-schmidtt/datascientest_mlops_project_weather.mlflow
```

**Tracked information:**
- **Parameters:** split_id, years, hyperparameters
- **Metrics:** F1, accuracy, precision, recall, ROC-AUC
- **Artifacts:** model, confusion matrix, classification report
- **Tags:** `is_production`, `split_id`, `years`, `archived`, `archived_at`

---

### 5. Experiment Organization on MLflow

#### **Manual Training**

- **Experiment:** `WeatherAUS_YearBased_Training`
- **Usage:** When calling `/train` API endpoint directly
- **Purpose:** All manual training runs grouped in this experiment

#### **Automated Pipeline**

- **Experiment:** `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`
- **Usage:** When calling `/pipeline` API endpoints
- **Important:** Experiment name is generated **once at API startup**
- **Behavior:** All splits from one automation run are grouped in the same experiment

**Example:**
```
API started at 10:30 → Experiment: 20260115_1030_Automated_Pipeline_...
  ├── Split 1
  ├── Split 2
  ├── Split 3
  └── ... Split 9
```

#### **Starting a New Experiment**

To create a new experiment with a fresh timestamp:

1. **Stop API:** Docker container usage: stop API container, local usage: `Ctrl+C`
2. **Clean splits:** local usage: `rm -rf data/automated_splits/split_*`
3. **Restart API:** Docker container usage: start API container, local usage:`python src/api/main.py`
4. **Result:** New experiment created with current timestamp

---
---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
    """, unsafe_allow_html=True)
