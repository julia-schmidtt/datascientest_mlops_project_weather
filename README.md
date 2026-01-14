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

Project Organization
------------

```
├── data/
│   ├── raw/                                 # Original raw data (weatherAUS.csv)
│   ├── interim/                             # Preprocessed data
│   ├── processed/                           # Train/test splits, filled missing values, One-Hot Encoded, scaled, ready for modeling
│   └── training_data_splits_by_year/        # Temporal splits of training data (2008, 2008+2009, 2008+2009+2010...), test data always the same
│
├── models/
│   ├── scaler.pkl                           # StandardScaler for API, extracted from preprocessing phase
│   └── xgboost_model_split_XX.pkl           # Trained models (not committed)
│
├── src/
│   ├── api/
│   │   ├── main.py                          # FastAPI application
│   │   ├── defaults.json                    # Training defaults for filling missing API inputs in simplied prediction endpoint, extracted from preprocessing phase
│   │   └── validation_data.json             # Locations & seasons for API input validation, extracted from preprocessing phase
│   ├── config/
│   │   └── __init__.py                      # Parameter management
│   ├── data/
│   │   ├── preprocess.py                    # Full data preprocessing: generated data in data/interim, processed and src/api/defaults.json, validation_data.json
│   │   └── training_data_splits_by_year.py  # Splits processed training data based on year, generates data in data/training_data_splits_by_year
│   └── models/
│       └── train_model.py                   # Model training with MLflow
│
├── tests/
│   └── test_api_prediction.py               # Script for full prediction API endpoint with 110 input features (random sample of test set is extracted and sent to production model for prediction)
│
├── params.yaml                              # Configuration parameters
├── requirements.txt                         # Python dependencies
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
```bash
dvc pull
```

This downloads:
- `data/raw/weatherAUS.csv`
- `data/interim/df_preprocessed.csv`
- `data/processed/X_train.csv, y_train.csv, X_test.csv, y_test.csv`
- `data/training_data_splits_by_year/` (9 temporal splits)

----------

Usage
----------

### Data Preprocessing

If you want to reprocess from scratch: put raw weatherAUS.csv in data/raw, then:

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

### Start API
```bash
python src/api/main.py
```

API runs on `http://localhost:8000`

## API Endpoints
### Health check
```bash
curl http://localhost:8000/health
```

### Model Info
```bash
curl http://localhost:8000/model/info
```

### Reload Model
```bash
curl -X POST http://localhost:8000/model/refresh
```

### Prediction
#### Simplified Prediction (5 required fields)
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

#### Full Prediction (110 features)
```bash
python tests/test_api_prediction.py
```

### Training
```bash
curl -X POST "http://localhost:8000/train?split_id=1"
```

----------

### XGBoost hyperparameters

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

----------

## Workflow Summary
```
1. Data → DVC versioning
2. Preprocess → Raw data preprocessing, remove and fill missing value after splitting in train and test set to prevent data leakage, scaler, defaults
3. Train → MLflow tracking, auto-promotion
4. API → Serve production model
5. Predict → Simplified or full endpoint
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
