#%%


import pandas as pd 
import os
import requests

# csv file URL from Huggingface repo
url = "https://huggingface.co/datasets/NUS-UAL/global-streetscapes/resolve/main/cities688.csv?download=true"
folder_path = "dataset"
file_name = "cities688.csv"
file_path = os.path.join(folder_path, file_name)

# Check if folder exists, if not create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)

# Check if file exists in the folder
if os.path.isfile(file_path):
    print(f"File '{file_name}' exists in '{folder_path}'.")
else:
    print(f"File '{file_name}' does not exist in '{folder_path}'. Downloading...")
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded '{file_name}' to '{folder_path}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        raise

#%%

# read the dataset using pandas 
dataset = pd.read_csv(filepath_or_buffer="dataset/cities688.csv")

#%%
# Display summary statistics of the dataset
print(dataset.describe())

# Display information about the dataset (data types, non-null values, etc.)
print("\nDataset Info:")
print(dataset.info())

# Display first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(dataset.head())
# %%

# Get all unique cities
print("All unique cities in the dataset:")
print(dataset['city'].unique())

# Find specific city (Gothenburg)
gothenburg_data = dataset[dataset['city'] == 'gothenburg']
print("\nData for Gothenburg:")
print(gothenburg_data)

# Check if Gothenburg exists in the dataset
if len(gothenburg_data) > 0:
    print("\nFound Gothenburg in the dataset!")
else:
    print("\nGothenburg not found in the dataset. Available cities might use a different spelling.")

# Find all cities that start with 'g' (case insensitive)
g_cities = dataset[dataset['city'].str.lower().str.startswith('g')]
print("\nCities that start with 'g':")
print(g_cities['city'].unique())
# %%