# --------------------------------------------------------------
# Calculating Outliers (and removing them)
# --------------------------------------------------------------

# Define a threshold for the Z-score to identify outliers (e.g., |Z-score| > 3)
z_score_threshold = 3

# Calculate Z-scores for the entire DataFrame
z_scores = np.abs((df3 - df3.mean()) / df3.std())

# Create a new column "Is_Outlier" for each row indicating whether any of the Z-scores exceed the threshold
df['Is_Outlier'] = z_scores.apply(lambda row: any(row > z_score_threshold), axis=1)

# Calculate the percentage of inliers and outliers
outlier_percentage = df['Is_Outlier'].value_counts(normalize=True) * 100

# Plotting the percentage of inliers and outliers
plt.figure(figsize=(12, 4))
outlier_percentage.plot(kind='barh', color='#ff6200')

# Adding the percentage labels on the bars
for index, value in enumerate(outlier_percentage):
    plt.text(value, index, f'{value:.2f}%', fontsize=15)

plt.title('Percentage of Inliers and Outliers')
plt.xticks(ticks=np.arange(0, 115, 5))
plt.xlabel('Percentage (%)')
plt.ylabel('Is Outlier')
plt.gca().invert_yaxis()
plt.show()


# Filter rows where 'Is_Outlier' is False (i8e. remove outliers)
df3 = df3[df3['Is_Outlier'] == False]
df3.reset_index(drop=True, inplace=True)
df3.shape
