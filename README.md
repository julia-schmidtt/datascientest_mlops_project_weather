# Rain Prediction in Australia - MLOps Project
----------

This project implements a MLOps pipeline for rain prediction in Australia, including:
- Data versioning with DVC
- Automated data preprocessing
- Automated training pipeline with year based training data splits (2008-2016)
- Experiment tracking with MLflow (XGBoost model is trained on scaled and SMOTE training data)
- Automated model comparison (new vs. production) and automated promotion of better model to production
- API with train and predict (full input data of 110 features, simplified input data with 5 required features) endpoints
- Additional API endpoints: health check, model information, model reload
- API pipeline endpoint (/pipeline/next-split): create training data split, data tracking with DVC, model training and tracking with MLflow, production model loading
- API pipeline endpoint(/pipeline/next-split-drift-detection): automated pipeline with drift monitoring (create split, track, drift check, conditional training)
- cronjob for scheduled automation for simulation of training data evolution over time and new training for each training data step

Project Organization
------------

```
├── data/
│   ├── raw/                                        # Original raw data (weatherAUS.csv)
│   ├── interim/                                    # Preprocessed data
│   ├── processed/                                  # Train/test splits, filled missing values, One-Hot Encoded, scaled, ready for modeling
│   ├── training_data_splits_by_year/               # Temporal splits (manual creation) of training data (2008, 2008+2009, 2008+2009+2010...), test data always the same
│   └── automated_splits/                           # Automated pipeline splits (DVC tracked)
│
├── models/
│   ├── scaler.pkl                                  # StandardScaler for API, extracted from preprocessing phase
│   └── xgboost_model_split_XX.pkl                  # Trained models (not committed)
│
├── monitoring/
│   └── reports/                                    # Evidently drift detection HTML reports
│
├── src/
│   ├── api/
│   │   ├── main.py                                 # FastAPI application
│   │   ├── defaults.json                           # Training defaults for filling missing API inputs in simplied prediction endpoint, extracted from preprocessing phase
│   │   └── validation_data.json                    # Locations & seasons for API input validation, extracted from preprocessing phase
│   ├── config/
│   │   └── __init__.py                             # Parameter management
│   ├── data/
│   │   ├── preprocess.py                           # Full data preprocessing: generated data in data/interim, processed and src/api/defaults.json, validation_data.json
│   │   ├── training_data_splits_by_year.py         # Splits processed training data based on year (manual), generates data in data/training_data_splits_by_year
│   │   └── automation_create_split.py              # Automated incremental training data splits
│   ├── monitoring/
│   │   └── data_drift.py                           # Data drift detection (Evidently)
│   └── models/
│       ├── train_model.py                          # Model training with MLflow
│       └── predict_model.py                        # Prediction script
│
├── scripts/
│   ├── process_next_split.sh                       # Cron job wrapper script
│   ├── process_next_split_with_drift_detection.sh  # Cron job wrapper script (with drift detection)
│   └── archive_all_models.py                       # Script to archive all models in registry to start without production model
│
├── logs/                                           # Pipeline execution logs
├── tests/
│   └── test_api_prediction.py                      # Script for full prediction API endpoint with 110 input features (random sample of test set is extracted and sent to production model for prediction)
│
├── params.yaml                                     # Configuration parameters
├── requirements.txt                                # Python dependencies
└── README.md                 
```

------------

Setup
------------
### 1. Clone Repository
```bash
git clone https://github.com/julia-schmidtt/datascientest_mlops_project_weather.git
cd datascientest_mlops_project_weather
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure DagsHub Credentials

Create `.env` file in project root:
```bash
nano .env
```

Add your DagsHub credentials:
```env
DAGSHUB_USERNAME=julia-schmidtt
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/julia-schmidtt/datascientest_mlops_project_weather.mlflow
```

**Get your DagsHub token:** DagsHub → Settings → Tokens 

### 5. Initialize DVC
```bash
dvc init
dvc remote add origin https://dagshub.com/your-username/datascientest_mlops_project_weather.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user your-username
dvc remote modify origin --local password your-token
```

### 6. Pull Data from DVC
**Option A: Minimal pull (only raw data)**

Pull only raw data and generate everything else locally:
```bash
# Pull only raw data
dvc pull data/raw/weatherAUS.csv.dvc

# Generate all preprocessing outputs
python src/data/preprocess.py

# Generate manually training splits
python src/data/training_data_splits_by_year.py
```


**Option B: Full pull**

Pull all tracked data from dvc.
```bash
dvc pull
```

**This downloads:**
- Raw data: data/raw/weatherAUS.csv
- Preprocessed data: data/interim/df_preprocessed.csv
- Preprocessed training/test splits (ready for modeling with full dataset): data/processed/X_train, y_train, X_test, y_test
- Manual temporal splits (9 year-based splits): data/training_data_splits_by_year/split_01_2008, split_02_2008-2009, ...
- Automatically created temporal splits during execution of automated ML pipeline: data/automated_splits/split_01_2008, ...


**IMPORTANT:** 
- Some files must be generated locally even after `dvc pull`:
```bash
# Generate scaler, defaults, and validation data
python src/data/preprocess.py
```
**This creates:**
- `models/scaler.pkl` - Required for input for  API predictions
- `src/api/defaults.json` - Default values for missing features in input for API predictions
- `src/api/validation_data.json` - Valid locations and seasons to check input for API prediction


- If you want to use the automated API endpoint from start to end  (pipeline/next-split) you have to remove pulled data in data/automated_splits/
```bash
# Remove old automation splits
rm -rf data/automated_splits/split_*
rm -f data/automated_splits/metadata.yaml

# Verify
ls -la data/automated_splits/  # Should only show .gitkeep
```

----------

Manual Usage
----------

### Data Preprocessing

If you want to reprocess from scratch: pull raw weatherAUS.csv from dvc (Pull Data from DVC - Option A), then:

```bash
python src/data/preprocess.py
```

**Output:**
- `data/interim/df_preprocessed.csv`
- `data/processed/X_train.csv, y_train.csv, X_test.csv, y_test.csv`
- `models/scaler.pkl`
- `src/api/defaults.json`
- `src/api/validation_data.json`


### Create Temporal Splits
```bash
python src/data/training_data_splits_by_year.py
```

**Output:** 9 cumulative year-based splits in `data/training_data_splits_by_year/`
- Split 1: 2008
- Split 2: 2008-2009
- ...
- Split 9: 2008-2016


### Train Model

Train on a specific split:
```bash
python src/models/train_model.py --split_id 1
```

1. Loads training data splits
2. SMOTE applied to training data for class balancing
3. Trains XGBoost model
4. Performance evaluation on test set
5. Comparison of performance  with current production model
6. Automated promotion to production if F1 score of new trained model is better than F1 score of current production model
7. Logs everything to MLflow/DagsHub

**Train all splits:**
```bash
for i in {1..9}; do
    python src/models/train_model.py --split_id $i
done
```


----------
DVC Pipeline
----------

### Run Complete Pipeline

Execute the entire ML pipeline with dependency tracking:
```bash
dvc repro
```

**This runs:**
1. `preprocess` - Generate processed data from raw data
2. `create_splits` - Create 9 temporal training splits
3. `train_split_1` - Train model on first split

**DVC automatically:**
- Tracks dependencies (code, data, params)
- Caches outputs
- Skips unchanged stages
- Ensures reproducibility

### Visualize Pipeline
```bash
# Show dependency graph
dvc dag

# Check pipeline status
dvc status

# View pipeline definition
cat dvc.yaml
```

### Re-run Specific Stage
```bash
# Only run preprocessing
dvc repro preprocess

# Force re-run (ignore cache)
dvc repro --force
```

**Note:** The DVC pipeline covers manual workflow. For automated continuous training, use the API pipeline endpoint `/pipeline/next-split` with cron jobs.


----------
Airflow Setup and Usage
----------
Add the following to you `.env` file:
```
AIRFLOW_UID=1000
AIRFLOW_GID=0
```
Make sure the `/dags`, `/logs` and `/plugins` directorys have the right read and write rights:
```
sudo chmod -R 777 logs/
sudo chmod -R 777 dags/
sudo chmod -R 777 plugins/
```
The Airflow container might take a while to start. After its up and healthy it runs on http://localhost:8080. The credentials are ```airflow``` for username and password. 
You should see the Dags ```process_next_split_dag``` and ```process_next_split_with_driftdetection_dag``` ready for execution.  


----------
Container Usage
----------
For the Container usage use the command ```docker compose up -d --build```. The first startup will take a while. The following Containers will be launched:
1. weather-fastapi on port 8000
2. Airflow and depndencies on port 8080
3. prometheus on port 9090
4. node-exporter on port 9100
5. grafana on port 3000

When working on the VM with VsCode it is possible, that you need to forward the ports to your machine to accsess. 
Shut down the containers after use ```docker compose down```
**Note:** Airflow is really resource hungry and will crash the normal VMs. Use the oen from the Airflow module. 

----------
Grafana
----------
After the startup of the cotnainer you can accsess via localhost http://localhost:3000. The password and username is admin.
When the Dashboard is not loaded you can import it on the Dashboard menu. Click new an choose import and drag and drop the Dashboard-json file from the directory /grafana/Dashboards. 
It can take some time for the metrics to show up and it helps to change the observed period. 



----------
API Usage
----------

### Start API
```bash
python src/api/main.py
```

API runs on `http://localhost:8000`

## API Endpoints
### Health Check Endpoint
```bash
curl http://localhost:8000/health | python -m json.tool
```

### Model Info Endpoint
```bash
curl http://localhost:8000/model/info | python -m json.tool
```

### Reload Model Endpoint
```bash
curl -X POST http://localhost:8000/model/refresh | python -m json.tool
```

### Prediction
#### Simplified Prediction (5 required fields) Endpoint
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Sydney",
    "date": "2025-01-15",
    "min_temp": 18.0,
    "max_temp": 28.0,
    "rain_today": 0
  }' | python -m json.tool
```

#### Full Prediction (110 features) Endpoint
```bash
python tests/test_api_prediction.py
```

### Training Endpoint
```bash
curl -X POST "http://localhost:8000/train?split_id=1" | python -m json.tool
```

### Automated Pipeline Endpoint
```bash
curl -X POST http://localhost:8000/pipeline/next-split | python -m json.tool
```
**Complete workflow:**
1. Create next temporal split
2. Track with DVC (`dvc add`, `git commit`, `git push`, `dvc push`)
3. Train XGBoost model on new split
4. Log to MLflow (metrics, parameters, artifacts)
5. Compare F1 score with production model
6. Auto-promote if better
7. Reload production model in API

**Output directory for generated training data split:** `data/automated_splits/`

**MLflow experiment naming:** `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`


### Automated Pipeline with Drift Detection Endpoint
```bash
curl -X POST http://localhost:8000/pipeline/next-split-drift-detection | python -m json.tool
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

**Drift Configuration (`params.yaml`):**
```yaml
monitoring:
  drift_check_enabled: true
  drift_threshold: 0.10  # change this parameter for threshold adjustment
```

**Drift Reports:** Saved to `monitoring/reports/`

**Output directory for generated training data split:** `data/automated_splits/`

**MLflow experiment naming:** `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`


----------
Cron Job Automation
----------

### Setup for continuous training of all data splits

#### 1. Archive all existing models in MLflow Model Registry and restart API
```bash
python scripts/archive_all_models.py
```
Archives all models in MLflow Registry to prepare for clean automation run.

```bash
python src/api/main.py
```

#### 2. Delete data in data/automated_splits to start with split 1 training
```bash
rm -rf data/automated_splits/split_*
rm -f data/automated_splits/metadata.yaml
ls -la data/automated_splits/
```
- only .gitignore should be in data/automated_splits after data removal


#### 3. Configure cronjob

**IMPORTANT:** Use absolute paths! Replace with your actual project directory.

**Find your absolute path:**
```bash
cd /path/to/your/project
pwd  # Copy this output
```

**Configure cron:**
```bash
crontab -e
```

**Option A: Pipeline WITHOUT Drift Detection (trains every split)**
```bash
# E.g. every 2 minutes 
*/2 * * * * /path/to/scripts/process_next_split.sh >> /path/to/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M).log 2>&1
```

**Option B: Pipeline WITH Drift Detection (conditional training)**
```bash
# E.g. every 2 minutes 
*/2 * * * * /path/to/scripts/process_next_split_with_drift_detection.sh >> /path/to/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M).log 2>&1
```

#### 4. Monitor Execution
```bash
# List cron jobs
crontab -l

# View logs (latest first)
ls -lth logs/
```

#### 5. Cron Workflow
**Option A**
```
Time 00:00: Split 1 created → Model v20 → Production 
Time 00:02: Split 2 created → Model v21 → Compare → Promote if better
Time 00:04: Split 3 created → Model v22 → Compare → Promote if better
...
Time XX:XX  Split 9 created → Model v28 → Compare → Promote if better
```

Each log file shows complete pipeline execution details including:
- Split creation info
- DVC/Git tracking status
- Model metrics (F1, accuracy, precision, recall, ROC-AUC)
- Promotion decision
- Production model version


**Option B**
```
Time 00:00: Split 1 created → Model v20 → Production 
Time 00:02: Split 2 created → Data Drift checked → If Data Drift above threshold → Model v21 → Compare performance with production model → Promote if better
...
```
----------

### XGBoost hyperparameters (`params.yaml`)

```yaml
max_depth: 9
learning_rate: 0.05
n_estimators: 200
colsample_bytree: 0.8
subsample: 0.7
gamma: 0.1
min_child_weight: 3
```

### SMOTE class balancing
**Original distribution:** 78% No Rain, 22% Rain

**After SMOTE:** 50% No Rain, 50% Rain

### Data Drift Monitoring (`params.yaml`)

- Uses **Evidently** for data drift detection
- Compares new training split against production model's training data
- Training triggered only if drift percentage exceeds threshold

## MLflow Integration
All experiments are tracked on **DagsHub MLflow**:
```
https://dagshub.com/julia-schmidtt/datascientest_mlops_project_weather.mlflow
```

**Tracked information:**
- Parameters (split_id, years, hyperparameters)
- Metrics (F1, accuracy, precision, recall, ROC-AUC)
- Artifacts (model, confusion matrix, classification report)
- Tags (`is_production`, `split_id`, `years`)


### Experiment Organization

**Manual Training:**
- Experiment: `WeatherAUS_YearBased_Training`
- Used when calling `/train` API endpoint directly
- All manual training runs grouped in this experiment

**Automated Pipeline:**
- Experiment: `YYYYMMDD_HHMM_Automated_Pipeline_WeatherPrediction_Australia`
- **Important:** Experiment name is generated **once at API startup**
- All splits from one automation run are grouped in the same experiment
- Example: API started at 10:30 → Experiment `20260115_1030_Automated_Pipeline_...`
  - Split 1, 2, 3, ..., 9 all logged to this experiment

**To start a new experiment run:**
1. Stop API (`Ctrl+C`)
2. Clean automated splits: `rm -rf data/automated_splits/split_*`
3. Restart API: `python src/api/main.py`
4. New experiment created with current timestamp

This design keeps all splits from one automation run together for easy comparison and tracking.

----------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
