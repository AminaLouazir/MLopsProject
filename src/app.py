import streamlit as st
import pandas as pd
import mlflow
from transformers import pipeline
import os
import yaml

# Load MLflow credentials
mlflow_config = yaml.safe_load(open(r"C:\Users\Lenovo\Documents\AISD\MLopsProject\credentials.yaml"))["mlflow_config"]
MLFLOW_TRACKING_URI = mlflow_config['MLFLOW_TRACKING_URI']
MLFLOW_TRACKING_USERNAME = mlflow_config['MLFLOW_TRACKING_USERNAME']
MLFLOW_TRACKING_PASSWORD = mlflow_config['MLFLOW_TRACKING_PASSWORD']

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the model
model_path = r"C:\Users\Lenovo\Documents\AISD\MLopsProject\models\zsmlc_classifier"
classifier = pipeline("zero-shot-classification", model=model_path)

# Define supported languages
supported_languages = ["English", "French", "German", "Spanish", "Russian"]

st.title("Multi-Language Sentiment Analysis")
st.write("Analyze sentiment across multiple languages using Zero-Shot Classification.")

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
