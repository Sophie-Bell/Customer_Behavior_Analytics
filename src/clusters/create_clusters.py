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

with open("../data/interim/02_customer_data_preprocessed.pkl", "rb") as file:
    df4 = pickle.load(file)

# --------------------------------------------------------------
# Standardize Data
# --------------------------------------------------------------

customer_ids = df4['CustomerID']


columns_to_scale = df4.columns.difference(['CustomerID'])

scaler = StandardScaler()  # Use StandardScaler

scaled_data = scaler.fit_transform(df4[columns_to_scale])

scaled_customer_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

# Re-add the "CustomerID" column
scaled_customer_df['CustomerID'] = customer_ids

scaled_customer_df = scaled_customer_df[['CustomerID'] + [col for col in scaled_customer_df.columns if col != 'CustomerID']]


# --------------------------------------------------------------
# Perform Principal Component Analysis (PCA)
# --------------------------------------------------------------

customer_ids = scaled_customer_df['CustomerID']

# Select the numeric columns (excluding "CustomerID")
numeric_columns = scaled_customer_df.columns.difference(['CustomerID'])

# Calculate the covariance matrix
cov_matrix = np.cov(scaled_customer_df[numeric_columns], rowvar=False)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues in descending order
sorted_indices = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Cross-validation to select the number of components
num_components_to_try = range(1, len(numeric_columns) + 1)
cumulative_variance_explained = []
for num_components in num_components_to_try:
    pca = PCA(n_components=num_components)
    pca.fit(scaled_customer_df[numeric_columns])
    variance_explained = sum(pca.explained_variance_ratio_)
    cumulative_variance_explained.append(variance_explained)


num_components = range(1, len(cumulative_variance_explained) + 1)

# PCA

pca = PCA(n_components=6)
reduced_data = pca.fit_transform(scaled_customer_df[numeric_columns])

# Creating a new dataframe from the PCA dataframe
reduced_data = pd.DataFrame(reduced_data, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

# Merge the "CustomerID" column back to the new PCA dataframe
reduced_data = pd.merge(scaled_customer_df[['CustomerID']], reduced_data, left_index=True, right_index=True)

# Reset the index in scaled_customer_df
scaled_customer_df.reset_index(inplace=True)

# Merge the "CustomerID" column back to the new PCA dataframe
reduced_data = pd.merge(scaled_customer_df[['CustomerID']], reduced_data, on='CustomerID')


# --------------------------------------------------------------
# Find Elbow Point
# --------------------------------------------------------------

# Plot the cumulative explained variance against the number of components
plt.figure(figsize=(10, 6))
plt.plot(num_components, cumulative_variance_explained, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)

# Find the elbow point
elbow_point = None
for i in range(1, len(cumulative_variance_explained)):
    if cumulative_variance_explained[i] - cumulative_variance_explained[i - 1] < 0.02: 
        elbow_point = i
        break

# Plot the elbow point
if elbow_point is not None:
    plt.scatter(elbow_point + 1, cumulative_variance_explained[elbow_point], color='red', label=f'Elbow Point: {elbow_point + 1}')

plt.legend()
plt.show()

# The elbow point is where you might choose the number of components
if elbow_point is not None:
    print(f'Elbow Point at {elbow_point + 1} components')


# --------------------------------------------------------------
# Perform K-Means Clustering
# -------------------------------------------------------------- 

# K = 3 looks to be best)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=0)
kmeans.fit(reduced_data)

# Get the frequency of each cluster
cluster_frequencies = Counter(kmeans.labels_)

# Create a mapping from old labels to new labels based on frequency
label_mapping = {label: new_label for new_label, (label, _) in
                 enumerate(cluster_frequencies.most_common())}

# Reverse the mapping to assign labels as per your criteria
label_mapping = {v: k for k, v in {2: 1, 1: 0, 0: 2}.items()}

# Apply the mapping to get the new labels
new_labels = np.array([label_mapping[label] for label in kmeans.labels_])

# Create a copy of the reduced_data DataFrame to avoid the SettingWithCopyWarning
df5 = reduced_data.copy()

# Add the 'Cluster' column to the DataFrame
df5['Cluster'] = new_labels


