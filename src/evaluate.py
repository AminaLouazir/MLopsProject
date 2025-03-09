import pandas as pd 
import numpy as np
import mlflow
import yaml
import os
from os.path import join
from transformers import pipeline

from packages.evaluate_utilities import get_data_sample, predictions_evaluation

# Get all the required yaml files
params_process = yaml.safe_load(open("params.yaml"))["preprocess"]
params_eval = yaml.safe_load(open("params.yaml"))["evaluate"]
mlflow_config = yaml.safe_load(open("credentials.yaml"))["mlflow_config"]

# Combined file path and reading the processed file
combined_file_path = join("data", "processed", params_process['final_file_name']+params_process['final_ext'])
combined_files = pd.read_csv(combined_file_path)

# Model definition
zsmlc_classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')

# Getting MLFlow credentials
MLFLOW_TRACKING_URI = mlflow_config['MLFLOW_TRACKING_URI']
MLFLOW_TRACKING_USERNAME = mlflow_config['MLFLOW_TRACKING_USERNAME']
MLFLOW_TRACKING_PASSWORD = mlflow_config['MLFLOW_TRACKING_PASSWORD']

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if __name__ == "__main__":

    # Set the experiment for MLFlow
    mlflow.set_experiment("Multi-linguage-classification")

    with mlflow.start_run():
        for language in combined_files['language'].unique():
            # Extract a sample for the given language
            lang_sample_data = get_data_sample(combined_files, language)

            # Get predictions and evaluation metrics
            pred_eval = predictions_evaluation(lang_sample_data, zsmlc_classifier)

            # Log different metrics for the language in MLFlow
            mlflow.log_metric(language + "_accuracy", pred_eval['accuracy'])
            mlflow.log_metric(language + "_f1_score", pred_eval['f1_score'])

            # Optionally, log other relevant metrics or parameters here
            # e.g., mlflow.log_param("parameter_name", "parameter_value")
            # mlflow.log_artifact("path_to_file") for artifacts/logs if needed

        # Save the model to a directory after the run is finished
        model_save_path = "models/zsmlc_classifier"
        zsmlc_classifier.save_pretrained(model_save_path)
