"""
Automated Split Creation for ML Pipeline

Creates next incremental training split for automated workflows.
This script is designed to be called by cron jobs or API endpoints.

Output directory: data/automated_splits/

Usage:
    python src/data/automation_create_split.py
    
Returns:
    Exit code 0 if successful, 1 if no more splits can be created
"""

import pandas as pd
import yaml
from pathlib import Path
import sys
from datetime import datetime


# Generate unique experiment name for automation runs
AUTOMATION_EXPERIMENT_NAME = f"Automated_Pipeline_WeatherPredictionAustralia_{datetime.now().strftime('%Y%m%d_%H%M')}"


# Function to create year based split from preprocessed training data
def create_next_split():
    """Create next temporal split for automation workflow"""

    print("Automated Training Data Split Creation")

    # Load processed data
    try:
        X_train_full = pd.read_csv('data/processed/X_train.csv')
        y_train_full = pd.read_csv('data/processed/y_train.csv')
        X_test_fixed = pd.read_csv('data/processed/X_test.csv')
        y_test_fixed = pd.read_csv('data/processed/y_test.csv')
    except FileNotFoundError as e:
        print(f"ERROR: Processed data not found: {e}")
        print("Run preprocessing first: python src/data/preprocess.py")
        return None, "Processed data not found"

    # Ensure Year column
    if 'Year' not in X_train_full.columns:
        print("ERROR: Year column not found!")
        return None, "Year column missing"
    
    X_train_full['Year'] = X_train_full['Year'].round().astype(int)

    # Filter to complete years
    mask = (X_train_full['Year'] >= 2008) & (X_train_full['Year'] <= 2016)
    X_train_full = X_train_full[mask].reset_index(drop=True)
    y_train_full = y_train_full[mask].reset_index(drop=True)
    
    years_available = sorted(X_train_full['Year'].unique())
    print(f"Available years: {years_available}")
    
    # Output directory for automated splits
    splits_dir = Path('data/automated_splits')
    splits_dir.mkdir(exist_ok=True, parents=True)
    
    # Check existing splits
    existing_splits = sorted([
        int(d.name.split('_')[1]) 
        for d in splits_dir.glob("split_*")
    ])
    
    print(f"Existing splits: {existing_splits if existing_splits else 'None'}")


    if not existing_splits:
        # No splits exist, create first one
        next_split_id = 1
        end_year = years_available[0]  
        print(f"Creating first split: Split {next_split_id} (year {end_year})")
    else:
        # Find next year to add
        latest_split_dir = list(splits_dir.glob(f"split_{max(existing_splits):02d}_*"))
        
        if not latest_split_dir:
            print("ERROR: Could not find latest split directory!")
            return None, "Latest split not found"

        # Extract end year from latest split
        latest_split_name = latest_split_dir[0].name
        if '-' in latest_split_name:
            latest_end_year = int(latest_split_name.split('_')[-1].split('-')[-1])
        else:
            latest_end_year = int(latest_split_name.split('_')[-1])
        
        print(f"Latest split ends at year: {latest_end_year}")

        # Next year
        try:
            next_year_idx = years_available.index(latest_end_year) + 1
        except ValueError:
            print(f"ERROR: Year {latest_end_year} not found in available years")
            return None, "Year not found"
        
        if next_year_idx >= len(years_available):
            print(f"INFO: All years already processed (up to {latest_end_year})")
            return None, "All splits complete"
        
        next_split_id = max(existing_splits) + 1
        end_year = years_available[next_year_idx]
        print(f"Creating next split: Split {next_split_id} (year {end_year})")

    # Create new split
    start_year = years_available[0] 
    
    mask = (X_train_full['Year'] >= start_year) & (X_train_full['Year'] <= end_year)
    X_train_split = X_train_full[mask].reset_index(drop=True)
    y_train_split = y_train_full[mask].reset_index(drop=True)

    # Split name
    if start_year == end_year:
        split_name = f"{end_year}"
    else:
        split_name = f"{start_year}-{end_year}"
    
    split_dir = splits_dir / f'split_{next_split_id:02d}_{split_name}'
    split_dir.mkdir(exist_ok=True)

    # Save data
    X_train_split.to_csv(split_dir / 'X_train.csv', index=False)
    y_train_split.to_csv(split_dir / 'y_train.csv', index=False)
    X_test_fixed.to_csv(split_dir / 'X_test.csv', index=False)
    y_test_fixed.to_csv(split_dir / 'y_test.csv', index=False)

    # Class distribution
    n_no_rain = (y_train_split['RainTomorrow'] == 0).sum()
    n_rain = (y_train_split['RainTomorrow'] == 1).sum()
    pct_no_rain = n_no_rain / len(y_train_split) * 100
    pct_rain = n_rain / len(y_train_split) * 100
    
    split_info = {
        'split_id': next_split_id,
        'years': split_name,
        'start_year': int(start_year),
        'end_year': int(end_year),
        'train_samples': len(X_train_split),
        'test_samples': len(X_test_fixed),
        'class_distribution': {
            'no_rain': float(pct_no_rain / 100),
            'rain': float(pct_rain / 100)
        }
    }
    

    # Update metadata
    metadata_path = splits_dir / 'metadata.yaml'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
    else:
        metadata = {
            'total_train_samples': len(X_train_full),
            'test_samples': len(X_test_fixed),
            'random_state': 42,
            'split_method': 'cumulative_by_year_automated',
            'years_available': [int(y) for y in years_available],
            'n_features': len(X_train_full.columns) - 1,
            'splits': []
        }
    
    metadata['splits'].append(split_info)

    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created Split {next_split_id:02d} ({split_name})")
    print(f"Directory: {split_dir}")
    print(f"Train samples: {len(X_train_split):,}")
    print(f"Test samples:  {len(X_test_fixed):,}")
    print(f"Class distribution:")
    print(f"    - No Rain: {pct_no_rain:.1f}%")
    print(f"    - Rain: {pct_rain:.1f}%")
    
    return next_split_id, split_info


if __name__ == "__main__":
    split_id, info = create_next_split()
    
    if split_id is None:
        print(f"INFO: {info}")
        if "All splits complete" in info:
            print("No more splits to create.")
            sys.exit(0)  # Not an error, just complete
        else:
            sys.exit(1)  # Actual error
    else:
        print(f"SUCCESS: Split {split_id} created successfully")
        sys.exit(0)

