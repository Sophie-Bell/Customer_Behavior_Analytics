# --------------------------------------------------------------
# Import Libraries
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import pickle
import seaborn as sns
import random
from scipy.stats import linregress

# --------------------------------------------------------------
# Load df
# --------------------------------------------------------------

with open("../data/interim/02_data_preprocessed.pkl", "rb") as file:
    df2 = pickle.load(file)

# --------------------------------------------------------------
# Days Since Last Purchase
# --------------------------------------------------------------

# Convert InvoiceDate to datetime type
df2['InvoiceDate'] = pd.to_datetime(df2['InvoiceDate'])

# Convert InvoiceDate to datetime and extract only the date
df2['InvoiceDay'] = df2['InvoiceDate'].dt.date

# Find the most recent purchase date for each customer
customer_data = df2.groupby('CustomerID')['InvoiceDay'].max().reset_index()

# Find the most recent date in the entire dataset
latest_purchase = df2['InvoiceDay'].max()

# Convert InvoiceDay to datetime type before subtraction
customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
latest_purchase = pd.to_datetime(latest_purchase)

# Calculate the number of days since the last purchase for each customer
customer_data['Days_Since_Last_Purchase'] = (latest_purchase - customer_data['InvoiceDay']).dt.days

# --------------------------------------------------------------
# Total Purchases
# --------------------------------------------------------------

total_transactions = df2.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
total_transactions.rename(columns={'InvoiceNo': 'Total_Purchases'}, inplace=True)

# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, total_transactions, on='CustomerID')


# --------------------------------------------------------------
# Total Spent
# --------------------------------------------------------------

# total amount spent
df2['Total_Spent'] = df2['UnitPrice'] * df2['Quantity']
total_spent = df2.groupby('CustomerID')['Total_Spent'].sum().reset_index()


# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, total_spent, on='CustomerID')

# --------------------------------------------------------------
# Total Products Bought
# --------------------------------------------------------------

# Calculate the total number of products purchased by each customer
total_products_purchased = df2.groupby('CustomerID')['Quantity'].sum().reset_index()
total_products_purchased.rename(columns={'Quantity': 'Total_Products_Bought'}, inplace=True)

customer_data = pd.merge(customer_data, total_products_purchased, on='CustomerID')

# --------------------------------------------------------------
# Average Value/Purchase (Average Value Per Purchase)
# --------------------------------------------------------------

average_transaction_value = total_spent.merge(total_transactions, on='CustomerID')
average_transaction_value['Average_Value/Purchase'] = average_transaction_value['Total_Spent'] / average_transaction_value['Total_Purchases']

# Merge the average transaction value with your customer data
customer_data = pd.merge(customer_data, average_transaction_value[['CustomerID', 'Average_Value/Purchase']], on='CustomerID')


# Round the 'Average_Value/Purchase' column to 2 decimal points
customer_data['Average_Value/Purchase'] = customer_data['Average_Value/Purchase'].round(2)

# Display the first few rows of the customer_data DataFrame
customer_data.head()

# --------------------------------------------------------------
# Cancellation Frequency
# --------------------------------------------------------------

# Calculate the total number of transactions made by each customer
total_transactions = df2.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

# Calculate the number of cancelled transactions for each customer
cancelled_transactions = df2[df2['Transaction_Status'] == 'Cancelled']
cancellation_freq = cancelled_transactions.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
cancellation_freq.rename(columns={'InvoiceNo': 'Cancellation_Frequency'}, inplace=True)

# Merge the Cancellation Frequency data into the customer_data dataframe
customer_data = pd.merge(customer_data, cancellation_freq, on='CustomerID', how='left')

# Replace NaN values with 0 (for customers who have not cancelled any transaction)
customer_data['Cancellation_Frequency'].fillna(0, inplace=True)

# --------------------------------------------------------------
# Avg Monthly Spending
# --------------------------------------------------------------

# Extract month and year from InvoiceDate
df2['Year'] = df2['InvoiceDate'].dt.year
df2['Month'] = df2['InvoiceDate'].dt.month


# Calculate monthly spending for each customer
monthly_spending = df2.groupby(['CustomerID', 'Year', 'Month'])['Total_Spent'].sum().reset_index()

# Calculate the average monthly spending for each customer
average_monthly_spending = monthly_spending.groupby('CustomerID')['Total_Spent'].mean().reset_index()
average_monthly_spending.rename(columns={'Total_Spent': 'Average_Monthly_Spending'}, inplace=True)


# --------------------------------------------------------------
# Spending Trend
# --------------------------------------------------------------

def calculate_trend(spend_data):
    if len(spend_data) > 1:
        x = np.arange(len(spend_data))
        slope, _, _, _, _ = linregress(x, spend_data)
        return slope
    else:
        return 0

spending_trends = monthly_spending.groupby('CustomerID')['Total_Spent'].apply(calculate_trend).reset_index()
spending_trends.rename(columns={'Total_Spent': 'Trend'}, inplace=True)