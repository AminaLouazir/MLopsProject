import os
import yaml
import subprocess
from yaml import safe_load

try:
    # Load DVC and model configuration from params.yaml
    params_path = r"C:\Users\Lenovo\Documents\AISD\MLopsProject\params.yaml"
    params = safe_load(open(params_path))
    
    # DVC Configuration
    dvc_params = params["dvc_config"]
    REMOTE_URL = dvc_params["DVC_REMOTE_URL"]
    USERNAME = dvc_params["USERNAME"]
    PASSWORD = dvc_params["PASSWORD"]
    
    # Model and Data Paths
    model_data_params = params["model_data_config"]
    MODEL_PATH = model_data_params["MODEL_PATH"]
    DATA_PATH = model_data_params["METRICS_PATH"]
    
    # Load MLflow configuration from credentials.yaml
    credentials_path = r"C:\Users\Lenovo\Documents\AISD\MLopsProject\credentials.yaml"
    if os.path.exists(credentials_path):
        credentials = safe_load(open(credentials_path))
        if "mlflow_config" in credentials:
            mlflow_config = credentials["mlflow_config"]
            MLFLOW_TRACKING_URI = mlflow_config.get('MLFLOW_TRACKING_URI', '')
            print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        else:
            print("Warning: mlflow_config not found in credentials.yaml")
            MLFLOW_TRACKING_URI = ""
    else:
        print("Warning: credentials.yaml not found")
        MLFLOW_TRACKING_URI = ""
    
    # Create directories if they don't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Initialize DVC if not already initialized
    if not os.path.exists(".dvc"):
        subprocess.run(["dvc", "init"])
        print("DVC initialized")
    
    # Set up DVC remote
    subprocess.run(["dvc", "remote", "add", "origin", REMOTE_URL])
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"])
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "user", USERNAME])
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "password", PASSWORD])
    
    # Try to pull existing data (may fail on first run, which is OK)
    try:
        subprocess.run(["dvc", "pull"], check=False)
        print("DVC pull completed")
    except Exception as e:
        print(f"DVC pull failed - this is expected on first run: {e}")
    
    # Add model and data directories to DVC if they exist and aren't already tracked
    if os.path.exists(MODEL_PATH):
        try:
            subprocess.run(["dvc", "add", MODEL_PATH], check=False)
            print(f"Added {MODEL_PATH} to DVC")
        except Exception as e:
            print(f"Note: Could not add {MODEL_PATH} to DVC: {e}")
    
    if os.path.exists(DATA_PATH):
        try:
            subprocess.run(["dvc", "add", DATA_PATH], check=False)
            print(f"Added {DATA_PATH} to DVC")
        except Exception as e:
            print(f"Note: Could not add {DATA_PATH} to DVC: {e}")
    
    print("Preparation completed successfully")
    
except Exception as e:
    print(f"Error during preparation: {e}")