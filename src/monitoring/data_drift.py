"""
Data Drift Monitoring using Evidently.

Monitors data drift using current production model's training data as reference.
Alerts when drift is detected and recommends retraining.
"""

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from datetime import datetime
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class DataDriftMonitor:
    def __init__(self):
        """
        Initialize Data Drift Monitor
        
        Uses current production model's training data as reference.
        Automatically updates reference when production model changes.
        """
        self.reports_dir = Path("monitoring/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Get current production model's split
        self.reference_split = self._get_production_split_id()
        
        # Load reference data from production model's training split
        split_dirs = list(Path("data/automated_splits").glob(
            f"split_{self.reference_split:02d}_*"
        ))
        
        if split_dirs:
            self.reference_data = pd.read_csv(f"{split_dirs[0]}/X_train.csv")
            print(f"\n Data Drift Monitor initialized")
            print(f"Production Model: Split {self.reference_split}")
            print(f"Reference samples: {len(self.reference_data)}")
        else:
            raise FileNotFoundError(
                f"Training data for Split {self.reference_split} not found!"
            )
        
        # Column mapping for Evidently
        self.column_mapping = ColumnMapping(
            prediction=None,
            numerical_features=self._get_numerical_features(),
            categorical_features=self._get_categorical_features()
        )
    
    def _get_production_split_id(self):
        """Get split_id from current production model in MLflow"""
        try:
            client = MlflowClient()
            model_name = "RainTomorrow_XGBoost"
            
            # Get production model version
            versions = client.search_model_versions(f"name='{model_name}'")
            prod_versions = [v for v in versions if v.current_stage == "Production"]
            
            if prod_versions:
                prod_version = prod_versions[0]
                run = client.get_run(prod_version.run_id)
                split_id = int(run.data.params.get("split_id", 1))
                return split_id
        except Exception as e:
            print(f"Could not get production model info: {e}")
            print(f"Using default: Split 1")
        
        return 1  # Default fallback
    
    def _get_numerical_features(self):
        """Get list of numerical feature columns"""
        numerical_cols = self.reference_data.select_dtypes(
            include=['float64', 'int64']
        ).columns.tolist()
        return numerical_cols
    
    def _get_categorical_features(self):
        """Get list of categorical feature columns"""
        categorical_cols = self.reference_data.select_dtypes(
            include=['object', 'category', 'bool', 'int32']
        ).columns.tolist()
        return categorical_cols
    
    def generate_baseline_report(self):
        """
        Generate baseline report for current production model.
        
        This creates a reference report showing the data quality
        of the production model's training data.
        """
        print(f"\n=== Generating Baseline Report ===")
        print(f"Production Model: Split {self.reference_split}\n")
        
        # Internal validation: split reference into two parts
        split_idx = int(len(self.reference_data) * 0.8)
        reference = self.reference_data.iloc[:split_idx]
        current = self.reference_data.iloc[split_idx:]
        
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"baseline_production_split{self.reference_split:02d}_{timestamp}.html"
        report.save_html(str(report_path))
        
        print(f"Baseline report saved: {report_path}")
        return report_path
    
    def check_drift(self, current_data: pd.DataFrame, save_report=True):
        """
        Check for data drift in production data
        
        Args:
            current_data: DataFrame with current production data
            save_report: Whether to save HTML report
            
        Returns:
            dict with drift metrics and alert status
        """
        print(f"\n=== Data Drift Check ===")
        print(f"Reference: Production Model (Split {self.reference_split})")
        print(f"Current data samples: {len(current_data)}\n")
        
        # Create drift report
        report = Report(metrics=[DataDriftPreset()])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"drift_check_{timestamp}.html"
            report.save_html(str(report_path))
            print(f"Report saved: {report_path}")
        
        # Extract metrics
        result = report.as_dict()
        metrics = result['metrics'][0]['result']
        
        drift_detected = metrics.get('dataset_drift', False)
        num_drifted = metrics.get('number_of_drifted_columns', 0)
        share_drifted = metrics.get('share_of_drifted_columns', 0.0)
        
        drift_info = {
            "timestamp": datetime.now().isoformat(),
            "reference_split": self.reference_split,
            "dataset_drift": share_drifted > 0,
            "number_of_drifted_columns": num_drifted,
            "share_of_drifted_columns": share_drifted,
            "alert": self._generate_alert(share_drifted >0, num_drifted, share_drifted),
        }
        
        # Print summary
        print(f"Dataset Drift (Evidently): {drift_detected}")
        print(f"Drifted Columns: {num_drifted} ({share_drifted*100:.1f}%)")
        
        if drift_info['alert']['severity'] != 'OK':
            print(f"\nALERT: {drift_info['alert']['severity']}")
            print(f"{drift_info['alert']['message']}")
            print(f"Recommendation: {drift_info['alert']['recommendation']}")

        elif num_drifted > 0:
            print(f"\nMinor drift detected ({share_drifted*100:.1f}%), below critical thresholds")

        else:
            print(f"\nNo significant drift detected")
        
        return drift_info
    
    def check_drift_for_split(self, split_id: int, save_report=True):
        """
        Check drift for a specific split compared to production model
        
        Args:
            split_id: Split number to check
            save_report: Whether to save HTML report
            
        Returns:
            dict with drift metrics and alert status
        """
        print(f"\n=== Drift Check: Split {split_id} vs Production (Split {self.reference_split}) ===\n")
        
        # Load split data
        split_dirs = list(Path("data/automated_splits").glob(f"split_{split_id:02d}_*"))
        
        if not split_dirs:
            # Try automated splits
            split_dirs = list(Path("data/automated_splits").glob(f"split_{split_id:02d}_*"))
        
        if not split_dirs:
            raise FileNotFoundError(f"Split {split_id} not found!")
        
        current_data = pd.read_csv(f"{split_dirs[0]}/X_train.csv")
        
        # Use the existing check_drift method
        drift_info = self.check_drift(current_data, save_report=save_report)
        
        # Add split_id to result
        drift_info['current_split'] = split_id
        
        return drift_info

    def _generate_alert(self, drift_detected, num_drifted, share_drifted):
        """
        Generate alert based on drift severity.

        Alert levels align with training threshold (from params.yaml):
        - INFO - Below training threshold, no action needed
        - WARNING - Training triggered because above threshold, moderate drift
        - CRITICAL - Training triggered, severe drift (2x threshold)
        
        Note: Alert severity is for monitoring/notification purposes.
        Training decision is made separately by API using params.yaml threshold.
        """
        
        if not drift_detected:
            return {
                "severity": "OK",
                "message": "No data drift detected",
                "recommendation": "Continue monitoring"
            }

        # Load training threshold from config
        try:
            import yaml
            with open('params.yaml', 'r') as f:
                params = yaml.safe_load(f)
                training_threshold = params.get('monitoring', {}).get('drift_threshold', 0.10)
        except:
            training_threshold = 0.1  # Fallback

        critical_threshold = training_threshold * 2.0  # 2x = critical
        
        if share_drifted >= critical_threshold:
            severity = "CRITICAL"
            recommendation = f"SEVERE DRIFT DETECTED (>{critical_threshold*100:.0f}%) - Training triggered, immediate investigation recommended."

        elif share_drifted >= training_threshold:
            severity = "WARNING"
            recommendation = f"DRIFT DETECTED (≥{training_threshold*100:.0f}%) - Training triggered, monitoring recommended."

        else:
            severity = "INFO"
            recommendation = f"MINOR DRIFT (<{training_threshold*100:.0f}%) - Below training threshold, continue monitoring."
        
        return {
            "severity": severity,
            "message": f"Data drift detected in {num_drifted} columns ({share_drifted*100:.1f}%)",
            "recommendation": recommendation
        }


if __name__ == "__main__":
    print("="*60)
    print("DATA DRIFT MONITORING - PRODUCTION MODE")
    print("="*60)
    
    # Initialize monitor
    monitor = DataDriftMonitor()
    
    # Generate baseline report
    monitor.generate_baseline_report()
    
    # Example: Check drift for a different split (simulating production data)
    print("\n" + "="*60)
    print("SIMULATION: Check if newer data has drifted")
    print("="*60)
    
    # Try to load a later split to simulate production drift
    try:
        # Load Split 9 as "production data" to check drift
        test_split_path = "data/training_data_splits_by_year/split_09_2008-2016/X_train.csv"
        test_data = pd.read_csv(test_split_path)
        
        # Sample random subset to simulate production data
        production_sample = test_data.sample(n=min(5000, len(test_data)), random_state=42)
        
        print(f"\nSimulating production data: {len(production_sample)} samples from Split 9")
        
        # Check drift
        drift_result = monitor.check_drift(production_sample)
        
    except FileNotFoundError:
        print("\nSplit 9 not found - cannot run simulation")
