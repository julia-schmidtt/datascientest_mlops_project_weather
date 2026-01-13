"""
Calculate default values (medians) from training data (after filling missing values, before scaling) for filling missing features in API prediction input data.
These API input values are unscaled and will be scaled with the saved StandardScaler during src/data/preprocess.py.

Usage:
    python src/api/prepare_defaults.py
"""

import pandas as pd
import json
from pathlib import Path

print("Calculating default values from training data.")


# Numerical features (calculate median)
numerical_cols = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'
]


defaults = {}
for col in numerical_cols:
    defaults[col] = float(X_train[col].median())


print("Default values (medians):")
for key, value in defaults.items():
    print(f"{key}: {value:.2f}")


# Save to JSON
output_dir = Path('src/api')
output_dir.mkdir(exist_ok=True)


with open(output_dir / 'defaults.json', 'w') as f:
    json.dump(defaults, f, indent=2)

print(f" Saved to {output_dir / 'defaults.json'}")


# Also extract all possible locations for validation
locations = [col.replace('Location_', '') for col in X_train.columns if col.startswith('Location_')]

validation_data = {
    'locations': locations,
    'seasons': ["Summer", "Autumn", "Winter", "Spring"]
}


with open(output_dir / 'validation_data.json', 'w') as f:
    json.dump(validation_data, f, indent=2)


print(f"Saved validation data to {output_dir / 'validation_data.json'}")
print(f"Locations: {len(locations)}")
print(f"Seasons: {len(validation_data['seasons'])}")



