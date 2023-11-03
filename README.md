# Customer-Segmentation-and-Analysis :dollar:
## Objective:
The primary objective of this project is to segment customers based on their purchasing behavior and provide actionable insights to enhance marketing strategies, optimize product offerings, and improve overall customer engagement. Based on my analysis, **I will segment customers into clusters based on their purchasing behavior and preferences.**

## Project Description :open_file_folder:
In this project, I will do the following:
- Explore the dataset to understand its structure and contents
- Clean the data, handle missing values
- Perform feature engineering to create new relevant features that enhances the analysis and clustering process
- Utilize clustering techniques to group customers with similar purchasing patterns, behaviors, and preferences
- Identify distinct customer segments that can provide valuable insights into the company's customer base
- Analyze customer behavior within each segment to uncover trends, such as preferred product categories, purchase frequency, and average order values

## Technologies Used  :computer:
<div style="display: flex; justify-content: space-between;">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="Pandas" width="40" height="40"/>
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="NumPy" width="40" height="40"/>
</div>

<br> <!-- Add a small space here -->

![Matplotlib](https://img.shields.io/badge/Matplotlib-3.1%2B-blueviolet?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.22%2B-yellow?style=flat-square)


## Data Source :bar_chart:
The **raw dataset** comes from the [E-Commerce Data Set](https://archive.ics.uci.edu/dataset/352/online+retail). This is a transnational data set that contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

This dataframe contains 8 variables that correspond to:


**InvoiceNo:** Invoice number. A 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. *(Nominal)*


**StockCode:** Product code. A 5-digit integral number uniquely assigned to each distinct product. *(Nominal)*


**Description:** Product name. *(Nominal)*


**Quantity:** The quantities of each product (item) per transaction. *(Numeric)*


**InvoiceDate:** Invoice Date and time, when each transaction was generated. *(Numeric)*


**UnitPrice:** Unit price. Product price per unit in sterling. *(Numeric)*


**CustomerID:** Customer number. A 5-digit integral number uniquely assigned to each customer. *(Nominal)*


**Country:** Country name of where each customer resides. *(Nominal)*


## Results :star:

Through comprehensive clustering analysis, our data was grouped into three distinct clusters, each characterized by unique customer behavior patterns. 

Some key findings from our analysis are:

**Cluster 0:** The characteristics of Cluster 0, such as the highest average value per purchase and monthly spending with a moderate total spending level, suggest a customer segment that values quality and is willing to spend more on individual items. This group might consist of high-end or luxury product enthusiasts who value exclusive and premium offerings. They are likely to appreciate personalized and high-value recommendations and could be a target for loyalty programs aimed at retaining their 
business and encouraging continued high-value purchases.

**Cluster 1:** Cluster 1 stands out for having the highest trend, total products bought, and total purchases among all clusters. This cluster may represent a segment of highly engaged and prolific customers. The high trend indicates growing customer loyalty and satisfaction. They could be a valuable focus for maintaining and expanding the customer base for a company. 

**Cluster 2:** Cluster 2 represents customers with the highest cancellation frequency, very high trend, and the highest number of days since the last purchase. This cluster's behavior could signify customers who have previously shown interest in the products but have encountered some issues or reservations. The high trend suggests that they might be open to re-engagement efforts. With moderate monthly spending and an average value per purchase, they may appreciate customized offers, improved customer service, and incentives to return. Addressing the cancellation frequency and resolving any underlying concerns could help rekindle their engagement

*These findings are valuable for tailoring marketing strategies, identifying customer preferences, and enhancing customer engagement for our business. They provide a foundation for targeted marketing campaigns, customer retention efforts, and personalized product recommendations, improving customer satisfaction.*
