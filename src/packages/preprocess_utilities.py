import pandas as pd 
from os import listdir
from os.path import join, splitext


def convert_tsv_to_pandas(file_name, path_raw):
  complete_path = join(path_raw, file_name)

  df = pd.read_csv(complete_path ,sep='\t')
  df.columns  = ["label", "text"]
  
  return df

def combine_all_files(path_raw, files_ext = '.tsv'):

  all_files = listdir(path_raw)
  list_all_dataframes = []

  for file_name in all_files:

    # Check the extention of the file
    if(file_name.endswith(files_ext)):      
      
      # Create pandas dataframe
      current_df = convert_tsv_to_pandas(file_name, path_raw)

      # Get the name of the file without the extension
      language  = splitext(file_name)[0] 
      current_df['language'] = str(language)
      
      list_all_dataframes.append(current_df)

  # Concate all the dataframes
  final_df = pd.concat(list_all_dataframes)

  return final_df

def data_preprocessor(raw_path, preprocessed_path, 
                      processed_file_name, 
                      files_ext = '.tsv', final_ext = '.csv'):
 
  # Create a single pandas dataframe for all the files 
  combined_pandas = combine_all_files(raw_path, files_ext)

  # Convert 0 1 to negative and positive and change the label column
  labels = {0: 'negative', 1: 'positive'}
  combined_pandas['label'] = combined_pandas['label'].map(labels)

  # Save the final file to destination
  complete_preprocessed_path = join(preprocessed_path, processed_file_name+final_ext)
  combined_pandas.to_csv(complete_preprocessed_path, encoding='utf-8', index=False)

  print("Preprocessing Complete!")