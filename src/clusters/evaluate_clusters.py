# --------------------------------------------------------------
# Import Libraries
# --------------------------------------------------------------

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
from datetime import datetime, timedelta
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from collections import Counter
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# --------------------------------------------------------------
# Open File
# --------------------------------------------------------------

with open("../data/interim/03_customer_data_preprocessed.pkl", "rb") as file:
    df = pickle.load(file)
    

# --------------------------------------------------------------
# Calculate Various Metrics
# --------------------------------------------------------------

# Compute number of customers
num_observations = len(df)

# Separate the features and the cluster labels
X = df.drop('Cluster', axis=1)
clusters = df['Cluster']

# Compute the metrics
sil_score = silhouette_score(X, clusters)
calinski_score = calinski_harabasz_score(X, clusters)
davies_score = davies_bouldin_score(X, clusters)

# Create a table to display the metrics and the number of observations
table_data = [
    ["Number of Observations", num_observations],
    ["Silhouette Score", sil_score],
    ["Calinski Harabasz Score", calinski_score],
    ["Davies Bouldin Score", davies_score]
]

# Print the table
for row in table_data:
    print(f"{row[0]}: {row[1]}")


# --------------------------------------------------------------
# Visualize Clusters
# --------------------------------------------------------------

#List of column names
attributes = ['Days_Since_Last_Purchase', 'Total_Purchases', 'Total_Spent', 'Average_Value/Purchase', 'Total_Products_Bought', 'Cancellation_Frequency', 'Trend', 'Average_Monthly_Spending']

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(merged_df[attributes])

# Create a figure with subplots for each cluster
fig, axes = plt.subplots(1, 3, figsize=(20, 10), subplot_kw=dict(polar=True))

# Cluster colors
colors = ['b', 'g', 'r']

# Number of clusters
n_clusters = 3

# Iterate over each cluster
for i, ax in enumerate(axes):
    # Data for the current cluster
    data = data_standardized[merged_df['Cluster'] == i].mean(axis=0).tolist()
    data += data[:1]  # Complete the loop

    # Set the cluster label as the title
    ax.set_title(f'Cluster {i}', color=colors[i], size=20, y=1.1)

    # Create angles for each axis
    angles = [n / float(len(attributes)) * 2 * pi for n in range(len(attributes))]
    angles += angles[:1]

    # Plot data
    ax.fill(angles, data, color=colors[i], alpha=0.25)

    # Set labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.set_yticklabels([])  

# Add a legend
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

plt.tight_layout()
plt.show()