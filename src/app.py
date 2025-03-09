import streamlit as st
import pandas as pd
import mlflow
from transformers import pipeline
import os
import yaml
from yaml import safe_load
from os.path import join

# Load MLflow credentials
mlflow_config = yaml.safe_load(open(r"C:\Users\Lenovo\Documents\AISD\MLopsProject\credentials.yaml"))["mlflow_config"]
MLFLOW_TRACKING_URI = mlflow_config['MLFLOW_TRACKING_URI']
MLFLOW_TRACKING_USERNAME = mlflow_config['MLFLOW_TRACKING_USERNAME']
MLFLOW_TRACKING_PASSWORD = mlflow_config['MLFLOW_TRACKING_PASSWORD']
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# DVC Configuration
def setup_dvc():
    try:
        # Get DVC parameters
        dvc_params = safe_load(open(r"C:\Users\Lenovo\Documents\AISD\MLopsProject\params.yaml"))["dvc_config"]
        REMOTE_URL = dvc_params["DVC_REMOTE_URL"]
        USERNAME = dvc_params["USERNAME"]
        PASSWORD = dvc_params["PASSWORD"]
        
        # Init DVC
        os.system("dvc init")
        
        # Set Authentication
        os.system(f"dvc remote add origin {REMOTE_URL}")
        os.system("dvc remote modify origin --local auth basic")
        os.system(f"dvc remote modify origin --local user {USERNAME}")
        os.system(f"dvc remote modify origin --local password {PASSWORD}")
        
        # Model and Data Parameters
        model_data_params = safe_load(open(r"C:\Users\Lenovo\Documents\AISD\MLopsProject\params.yaml"))["model_data_config"]
        MODEL_PATH = model_data_params["MODEL_PATH"]
        DATA_PATH = model_data_params["METRICS_PATH"]
        
        # Add model and data then Push to DagsHub Storage
        os.system(f"dvc add {MODEL_PATH} {DATA_PATH}")
        os.system("dvc push")
        
        return True, "DVC setup completed successfully."
    except Exception as e:
        return False, f"DVC setup failed: {str(e)}"

# Load the model
model_path = r"C:\Users\Lenovo\Documents\AISD\MLopsProject\models\zsmlc_classifier"
classifier = pipeline("zero-shot-classification", model=model_path)

# Define supported languages
supported_languages = ["English", "French", "German", "Spanish", "Russian"]

# Streamlit UI
st.title("Multi-Language Sentiment Analysis")
st.write("Analyze sentiment across multiple languages using Zero-Shot Classification.")

# DVC setup section
with st.expander("DVC Configuration"):
    if st.button("Setup DVC"):
        success, message = setup_dvc()
        if success:
            st.success(message)
        else:
            st.error(message)


# Select language
language = st.selectbox("Choose a language:", supported_languages)

# Get MLflow metrics
with mlflow.start_run():
    try:
        experiment = mlflow.get_experiment_by_name("Multi-linguage-classification")
        run = mlflow.search_runs(experiment_ids=[experiment.experiment_id]).iloc[-1]  # Get latest run
        accuracy = run[f"metrics.{language.lower()}_accuracy"]
        f1_score = run[f"metrics.{language.lower()}_f1_score"]
    except:
        accuracy, f1_score = "N/A", "N/A"

st.metric("Model Accuracy", accuracy)
st.metric("F1 Score", f1_score)

# Text input for sentiment analysis
text_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if text_input:
        labels = ["Positive", "Neutral", "Negative"]
        result = classifier(text_input, labels)
        st.write("Predicted Sentiment:", result["labels"][0])
    else:
        st.warning("Please enter text to analyze.")



