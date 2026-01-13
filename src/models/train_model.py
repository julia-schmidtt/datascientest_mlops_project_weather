"""
Train XGBoost model for rain prediction in Australia.
XGBoost was found as the best suited model for this binary classification task using LazyCLassifier on the cleaned dataset.
XGBoost parameters were identified using GridSearch.

This script loads year-based splits, applies SMOTE on training data for class balancing, trains an XGBoost classifier, 
saves the model, tracks everything with MLFlow, compares performance of new model with production model, 
automatically choose better model.

Input:  data/training_data_splits_by_year/split_XX
Output: MLFlow tracked model

Usage: python src/models/train_model.py --split_id 1
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import sys
from pathlib import Path
import argparse
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os


# Load environment variables FIRST
load_dotenv()


# Set MLflow credentials from .env (DagsHub authentication)
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN', '')


# Import params from params.yaml
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PARAMS


# MLFlow configuration from params.yaml
MLFLOW_TRACKING_URI = PARAMS['mlflow']['tracking_uri']
EXPERIMENT_NAME = PARAMS['mlflow']['experiment_name']
MODEL_NAME = "RainTomorrow_XGBoost"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load train data
# Load training and test set for specific split
def load_split_data(split_id: int):
    splits_dir = Path("data/training_data_splits_by_year")
    split_dirs = list(splits_dir.glob(f"split_{split_id:02d}_*"))

    if not split_dirs:
        raise FileNotFoundError(
            f"Split {split_id} not found in {splits_dir}!\n"
            f"Run: python src/data/training_data_splits_by_year.py"
        )

    split_dir = split_dirs[0]
    split_name = split_dir.name

    X_train = pd.read_csv(split_dir / "X_train.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_test = pd.read_csv(split_dir / "y_test.csv")

    year_info = split_name.split('_')[-1]
    
    print(f"Loaded Split {split_id}: {split_name}")
    print(f"Train: {len(X_train):6d} samples")
    print(f"Test:  {len(X_test):6d} samples")

    split_info = {
        'split_id': split_id,
        'split_name': split_name,
        'years': year_info,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return X_train, X_test, y_train, y_test, split_info


# Get current production model from MLFlow Registry 
def get_current_production_model():
    client = MlflowClient()
    
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if prod_versions:
            latest_prod = max(prod_versions, key=lambda x: int(x.version))
            run = client.get_run(latest_prod.run_id)
            f1_score_prod = run.data.metrics.get("f1_score", 0.0)
            return latest_prod, f1_score_prod
        
        return None, 0.0
    
    except Exception as e:
        print(f"No production model found: {e}")
        return None, 0.0


# Training function
def train_model(split_id=None):
    print(f"Training on split {split_id} (MLflow mode)")

    # ==================== Step 1: Load Data ====================
    X_train, X_test, y_train, y_test, split_info = load_split_data(split_id)

    # ==================== Step 2: Get Current Production Model ====================
    current_model_version, current_f1 = get_current_production_model()

    if current_model_version:
        print(f"Current Production Model:")
        print(f"Version: {current_model_version.version}")
        print(f"F1 Score: {current_f1:.4f}")
    elif split_id:
        print(f"No production model yet, this will be the first model.")

    # ==================== Step 3: Start MLflow Run ====================
    if split_id:
        run_name = f"split_{split_id:02d}_{split_info['years']}"
        mlflow_run = mlflow.start_run(run_name=run_name)
        print(f"MLflow Run Started: {mlflow_run.info.run_id}")
        
        # Log split info
        mlflow.log_param("split_id", split_id)
        mlflow.log_param("split_name", split_info['split_name'])
        mlflow.log_param("years", split_info['years'])
        mlflow.log_param("train_samples_original", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
    else:
        mlflow_run = None
    
    try:
        # ==================== Step 4: Apply SMOTE on training data ====================
        class_counts_before = y_train['RainTomorrow'].value_counts().to_dict()
        print('Applying SMOTE on training data.')
        print(f'Before: {class_counts_before}')
        
        smote = SMOTE(random_state=PARAMS['data']['random_state'])
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train.values.ravel())
        
        class_counts_after = pd.Series(y_train_smote).value_counts().to_dict()
        print(f'After: {class_counts_after}')
        
        mlflow.log_param("train_samples_after_smote", len(X_train_smote))
        mlflow.log_param("smote_applied", True)

        # ==================== Step 5: Train XGBoost Model ====================
        print('Training XGBoost Model.')

        model_params = {
            'max_depth': PARAMS['model']['max_depth'],
            'learning_rate': PARAMS['model']['learning_rate'],
            'n_estimators': PARAMS['model']['n_estimators'],
            'colsample_bytree': PARAMS['model']['colsample_bytree'],
            'subsample': PARAMS['model']['subsample'],
            'gamma': PARAMS['model']['gamma'],
            'min_child_weight': PARAMS['model']['min_child_weight'],
            'random_state': PARAMS['model']['random_state'],
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        print(f'Model parameters: {model_params}')

        if mlflow_run:
            mlflow.log_params(model_params)
        
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train_smote, y_train_smote)

        print('XGBoost model trained.')

        # ==================== Step 6: Evaluate ====================
        print('Model evaluation on test data.')
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }

        if mlflow_run:
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix and report
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_text(str(cm), "confusion_matrix.txt")
            
            report = classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'])
            mlflow.log_text(report, "classification_report.txt")
        
        print(f'\n Metrics:')
        for metric_name, value in metrics.items():
            print(f'{metric_name:12s}: {value:.4f}')

        # ==================== Step 7: Model Comparison ====================
        new_f1 = metrics["f1_score"]
        is_better = new_f1 > current_f1
        
        if mlflow_run and current_model_version:
            improvement = ((new_f1 - current_f1) / current_f1 * 100) if current_f1 > 0 else 100.0
            
            mlflow.log_metric("f1_vs_production", new_f1 - current_f1)
            mlflow.log_metric("improvement_percentage", improvement)
            mlflow.log_param("is_better_than_production", is_better)
            
            print("Model Comparison:")
            print(f"Production (v{current_model_version.version}): F1 = {current_f1:.4f}")
            print(f"New Model: F1 = {new_f1:.4f}")
            print(f"Improvement: {improvement:+.2f}%")
            print(f"{'New model performs better.' if is_better else 'Current model performs better.'}")
        elif mlflow_run:
            is_better = True
            print("First model - will become production")

        # ==================== Step 8: Log & Register Model ====================
        if mlflow_run:
            print(f"Logging model to MLflow.")
            
            signature = mlflow.models.infer_signature(X_train, y_pred)
            
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            
            model_uri = f"runs:/{mlflow_run.info.run_id}/model"

            print("Registering model in Model Registry.")

            model_details = mlflow.register_model(model_uri, MODEL_NAME)
            
            client = MlflowClient()
            tags = {
                "split_id": str(split_id) if split_id else "legacy",
                "years": split_info['years'] if split_info else "all",
                "f1_score": f"{new_f1:.4f}",
                "train_samples": str(len(X_train)),
            }

            # set all standard tags
            for key, value in tags.items():
                client.set_model_version_tag(MODEL_NAME, model_details.version, key, value)

            client.set_model_version_tag(MODEL_NAME, model_details.version, "is_production", "False")
            
            print(f"Registered as version {model_details.version}")

            # ==================== Step 9: Promote to Production? ====================
            if is_better:
                print("Promoting to production.")
                
                if current_model_version:
                    client.transition_model_version_stage(
                        MODEL_NAME,
                        current_model_version.version,
                        "Archived"
                    )
                    client.set_model_version_tag(MODEL_NAME, current_model_version.version, "is_production", "False")
                    print(f"Archived old version: {current_model_version.version}")
                
                client.transition_model_version_stage(
                    MODEL_NAME,
                    model_details.version,
                    "Production"
                )
                client.set_model_version_tag(MODEL_NAME, model_details.version, "is_production", "True")
                print(f"New production version: {model_details.version} (tagged is_production=True)")
            else:
                print(f"Keeping current production (v{current_model_version.version})")
                print(f"New model registered as v{model_details.version} but not promoted")

        # ==================== Step 10: Save as Pickle ====================
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f'xgboost_model_split_{split_id:02d}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f'Model saved: {model_path}')
        
        return new_f1, is_better
        
    finally:
        if mlflow_run:
            mlflow.end_run()        


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Train XGBoost model with MLflow tracking"
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        help="Split ID to train on (1-9)."
    )
    args = parser.parse_args()
    
    if args.split_id and not (1 <= args.split_id <= 9):
        print(f"ERROR: split_id must be between 1 and 9, got {args.split_id}")
        sys.exit(1)

    try:
        f1, promoted = train_model(split_id=args.split_id)
        
        print('Training complete.')
        print(f"F1 Score: {f1:.4f}")
        if args.split_id:
            print(f"Promoted: {'Yes' if promoted else 'No'}")
            print(f"MLflow UI: {MLFLOW_TRACKING_URI}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

