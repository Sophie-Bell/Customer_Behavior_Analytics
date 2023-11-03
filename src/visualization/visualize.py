# --------------------------------------------------------------
# Import Libraries
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

with open("../data/interim/02_customer_data_preprocessed.pkl", "rb") as file:
    df = pickle.load(file)

# --------------------------------------------------------------
# Box Plot Visualization
# --------------------------------------------------------------

def plot_multiple_boxplots(dataframe, column_names):
    # Check if any of the specified columns are not found in the DataFrame
    for column_name in column_names:
        if column_name not in dataframe.columns:
            print(f"Column '{column_name}' not found in the DataFrame.")
            return

    num_plots = len(column_names)
    num_rows = num_plots // 2 + num_plots % 2
    num_cols = 2

    plt.figure(figsize=(12, 8))
    for i, column_name in enumerate(column_names):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(data=dataframe, y=column_name)
        plt.title(f'Box Plot of {column_name}')
        plt.ylabel(column_name)

    plt.tight_layout()
    plt.show()
    
    
# --------------------------------------------------------------
# Correlation Matrix
# --------------------------------------------------------------

def plot_correlation_matrix(dataframe):
    # Calculate the correlation matrix
    corr_matrix = dataframe.corr()

    # Set the size of the heatmap
    plt.figure(figsize=(10, 8))

    # Create a heatmap using seaborn
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Set the title
    plt.title('Correlation Matrix')

    # Display the plot
    plt.show()
    
  
# --------------------------------------------------------------
# Correlation Matrix
# --------------------------------------------------------------  
    
def plot_pairplot(dataframe, numerical_columns):
    # Create a subset of the DataFrame with the specified numerical columns
    df_subset = dataframe[numerical_columns]

    # Create and display the pairplot
    sns.pairplot(df_subset, height=2)
    plt.show()
    
    
    
