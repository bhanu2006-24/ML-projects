# Wine Quality Prediction

This project focuses on predicting the quality of red wine using machine learning techniques. It utilizes the Red Wine Quality dataset to classify wines into "Good" or "Bad" quality based on physicochemical properties.

## 🔗 Project Notebook

[**View Colab Notebook**](https://colab.research.google.com/drive/1yqgI1tUqgL89I4z_hK5G6C8VjXOX4I-v)

## 📊 Dataset

The dataset used in this project is the **Red Wine Quality** dataset (Cortez et al., 2009).

- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- **Features:** 11 physicochemical properties (e.g., fixed acidity, volatile acidity, alcohol, etc.)
- **Target:** `quality` (score between 0 and 10)

## 🛠️ Technologies Used

- **Python**
- **Pandas** & **NumPy** (Data Manipulation)
- **Matplotlib** & **Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning)

## ⚙️ Project Workflow

1.  **Data Collection:** Downloading the dataset using `kagglehub`.
2.  **Data Preprocessing:**
    - Checking for missing values.
    - **Label Binarization:** Converting the `quality` score into a binary classification task:
      - **1 (Good Quality):** Quality score $\ge$ 7
      - **0 (Bad Quality):** Quality score < 7
3.  **Exploratory Data Analysis (EDA):**
    - Visualizing correlations between features and quality.
    - Heatmap for feature correlation matrix.
4.  **Model Training:**
    - Splitting data into training and testing sets (80-20 split).
    - Training a **Random Forest Classifier**.
5.  **Model Evaluation:**
    - Evaluating the model using Accuracy Score.
    - **Achieved Accuracy:** ~93.4%
6.  **Predictive System:**
    - Building a system to predict wine quality from new input data.

## 📈 Key Insights

- **Alcohol** content and **Sulphates** have a positive correlation with wine quality.
- **Volatile Acidity** has a negative correlation with wine quality.

## 🚀 How to Run

1.  Open the provided [Colab Notebook](https://colab.research.google.com/drive/1yqgI1tUqgL89I4z_hK5G6C8VjXOX4I-v).
2.  Run the cells sequentially to download the data, train the model, and see predictions.
