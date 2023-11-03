# --------------------------------------------------------------
# Import Libraries
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

df = pd.read_excel("../../data_exploration/raw/Online_Retail.xlsx")

# --------------------------------------------------------------
# Find Null Values
# --------------------------------------------------------------

for col in df.columns:
    print(f"{col}: {df[col].isnull().sum()}")

# --------------------------------------------------------------
# Find Unique Stock Codes
# --------------------------------------------------------------

unique_stock_codes = df['StockCode'].unique()

# Function to check if a code is "weird" (contains 0 or 1 digit)
def is_weird(code):
    return sum(c.isdigit() for c in str(code)) in (0, 1)

# Filter the unique stock codes to find the "weird" ones
anomalous_stock_codes = [code for code in unique_stock_codes if is_weird(code)]

# Printing each "weird" stock code and its frequency
print("Weird stock codes:")
for code in anomalous_stock_codes:
    frequency = (df['StockCode'] == code).sum()
    print(f"{code}: {frequency}")


# --------------------------------------------------------------
# Clean up the description column
# --------------------------------------------------------------

# Remove color words from the description for better cluster
def remove_color_words(description):
    color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white','cream']
    words = description.split()
    filtered_words = [word for word in words if word.lower() not in color_words]
    cleaned_description = ' '.join(filtered_words)
    return cleaned_description

# Assuming the description column is named 'Description'
df['Description'] = df['Description'].astype(str)

# Apply the remove_color_words function to the 'Description' column
df['Description'] = df['Description'].apply(remove_color_words)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

# upload to the interim 
df.to_pickle("../data/interim/01_data_preprocessed.pkl")