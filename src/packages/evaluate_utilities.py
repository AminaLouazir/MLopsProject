import pandas as pd 
import numpy as np
import yaml
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score


def make_prediction(clf_result):

  # Get the index of the maximum probability score
  max_index = np.argmax(clf_result["scores"])
  predicted_label = clf_result["labels"][max_index]

  return predicted_label


def run_batch_prediction(original_data, my_classifier, label_column='label', text_column = 'text'):

  # Make a copy of the data
  data_copy = original_data.copy()

  # The list that will contain the models predictions
  final_list_labels = []

  for index in range(len(original_data)):
    # Run classification
    sequences = original_data.iloc[index][text_column]
    candidate_labels = list(original_data[label_column].unique())
    result = my_classifier(sequences, candidate_labels, multi_class = True)

    # Make prediction
    final_list_labels.append(make_prediction(result))
  
  # Create the new column for the predictions
  data_copy["model_labels"] = final_list_labels

  return data_copy

def get_data_sample(df, language, sample_size = 100):

  lang_df = df[df['language']==language].sample(sample_size)

  return lang_df


def get_performance(df):

  performance = {}

  performance["accuracy"] = accuracy_score(df["label"], df["model_labels"])

  report = classification_report(df["label"], df["model_labels"], output_dict=True)
  
  performance["f1_score"] = report['macro avg']["f1-score"]

  return performance

def predictions_evaluation(data, my_classifier):

  predictions = run_batch_prediction(data, my_classifier)

  return get_performance(predictions)