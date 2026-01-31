# Customer Segmentation

This project uses unsupervised machine learning to segment mall customers based on their annual income and spending score. By identifying distinct customer groups, businesses can tailor their marketing strategies to target specific audiences more effectively.

## 🔗 Project Notebook

[View the Notebook](./Customer_Segmentation.ipynb) | [Open in Colab](https://colab.research.google.com/drive/1IFjJzlG-SCSz3GHHRBGI7k4cZYkorshJ?usp=sharing)

## 📊 Dataset

The dataset contains information about mall customers:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Customer's gender.
- **Age**: Customer's age.
- **Annual Income (k$)**: Annual income of the customer.
- **Spending Score (1-100)**: Score assigned by the mall based on customer behavior and spending nature.

The analysis specifically focuses on **Annual Income** and **Spending Score** for clustering.

## 🛠 Technologies Used

- **Python**
- **Pandas**: Data manipulation and DataFrame management.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning (K-Means Clustering).

## 🚀 Project Workflow

1.  **Data Loading**: Importing the Mall Customers dataset.
2.  **Data Exploration**: Checking dataset structure, shape, and missing values.
3.  **Feature Selection**: Selecting 'Annual Income' and 'Spending Score' for clustering.
4.  **Optimal Clusters**: Using the **Elbow Method** and WCSS (Within Clusters Sum of Squares) to determine the optimal number of clusters (which is 5).
5.  **Model Training**: Applying the **K-Means Clustering** algorithm with `k=5`.
6.  **Visualization**: Plotting the 5 customer clusters and their centroids.

## 📈 Key Insights & Results

The K-Means algorithm identified **5 distinct customer segments**:

1.  **High Income, Low Spending**: Target for exclusive offers or value proposition marketing.
2.  **Average Income, Average Spending**: Standard customers.
3.  **Low Income, Low Spending**: Budget-conscious shoppers.
4.  **Low Income, High Spending**: Potential target for affordable luxury or impulse buys.
5.  **High Income, High Spending**: VIP customers, primary target for premium products.

## 🧠 Key Concepts & Learnings

- **Unsupervised Learning**: Grouping data without predefined labels.
- **K-Means Clustering**: Partitioning n observations into k clusters.
- **Elbow Method**: A heuristic used in determining the number of clusters in a data set.
- **Cluster Visualization**: Visualizing high-dimensional data in 2D space.

## 💻 Setup & Usage

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Run the notebook `Customer_Segmentation.ipynb`.
