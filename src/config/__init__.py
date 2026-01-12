"""Configuration management for the project"""

import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_params():
    """Load parameters from params.yaml"""
    params_path = PROJECT_ROOT / "params.yaml"
    
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")
    
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

# Load parameters
PARAMS = load_params()

# MLflow configuration
MLFLOW_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    PARAMS['mlflow']['tracking_uri']
)

# DagsHub credentials (from .env file)
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', 'julia-schmidtt')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')

# Export for use in other modules
__all__ = [
    'PARAMS',
    'PROJECT_ROOT',
    'MLFLOW_URI',
    'DAGSHUB_USERNAME',
    'DAGSHUB_TOKEN'
]
