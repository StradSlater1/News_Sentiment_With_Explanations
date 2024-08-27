import pandas as pd
import numpy as np
import ast
from ast import literal_eval
import os
import glob


# Function to get all files from a folder
def get_csv_filenames(folder_path):
    # Get a list of all CSV files in the folder
    file_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(file_pattern)

    # Extract filenames from the file paths
    csv_filenames = [os.path.basename(file_path) for file_path in csv_files]

    return csv_filenames


# Import all the segmented labeled data files
files = get_csv_filenames('label_data')

# Create an empty df
dfs = pd.DataFrame()

# Combine segmented data into one df
for file in files:
    df = pd.read_csv(f'Data/{file}')
    dfs = pd.concat([dfs, df], axis=0)

# Drop index column
dfs = dfs.drop(columns='Unnamed: 0')

# Reset index
dfs.reset_index(inplace=True)

# Drop old index
dfs = dfs.drop(columns='index')

# Make each sentence encoding read as a list
dfs['Sentence Encodings'] = dfs['Sentence Encodings'].apply(ast.literal_eval)


# Extracts the phrases (keys) from the encodings
def extract_phrases(row):
    new_keys = []
    for key in row.keys():
        new_keys.append(key)
    return new_keys


# Extracts the values from the encodings
def extract_values(row):
    new_values = []
    for value in row.values():
        if value == '':
            new_values.append(0)
        elif int(value) == 0 or int(value) == 1:
            new_values.append(value)
        else:
            new_values.append(0)
    return new_values


# Creates values and phrase columns for each sentence
dfs['Phrases'] = dfs['Sentence Encodings'].apply(lambda x: extract_phrases(x))
dfs['Values'] = dfs['Sentence Encodings'].apply(lambda x: extract_values(x))

#dfs.to_csv('encoding_data.csv')