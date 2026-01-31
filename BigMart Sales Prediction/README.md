# Big Mart Sales Prediction

This project focuses on forecasting the sales of various products across different Big Mart outlets. By analyzing product attributes and store characteristics, an **XGBoost Regressor** model is trained to predict future sales, offering valuable insights for inventory optimization and sales strategies.

## 🔗 Project Notebook

[View the Notebook](./Big_Mart_Sales_Prediction.ipynb) | [Open in Colab](https://colab.research.google.com/drive/1L-Nq8c6aT5vdCyrItX4dLml4UC0bAs1m?usp=sharing)

## 📊 Dataset

The dataset comprises transaction records with the following key attributes:

- **Features**: `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`, `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`.
- **Target**: `Item_Outlet_Sales`.
- **Size**: 8523 records.

### Data Preprocessing:

- **Missing Values**: Imputed `Item_Weight` with the mean and `Outlet_Size` with the mode.
- **Data Standardization**: Harmonized inconsistent `Item_Fat_Content` labels (e.g., replacing 'LF' and 'low fat' with 'Low Fat').
- **Encoding**: Converted categorical variables into numeric format using **Label Encoding**.

## 🛠 Technologies Used

- **Python**
- **Pandas & NumPy**: For data cleaning and manipulation.
- **Matplotlib & Seaborn**: For exploratory data analysis and visualization.
- **Scikit-learn**: For data splitting and preprocessing.
- **XGBoost**: For building the regression model.

## 🚀 Project Workflow

1.  **Data Loading & Inspection**: Analyzing the dataset structure and types.
2.  **Data Cleaning**: Addressing missing data and correcting categorical inconsistencies.
3.  **Exploratory Data Analysis (EDA)**: Visualizing feature distributions and relationships.
4.  **Feature Engineering**: Transforming categorical data for model compatibility.
5.  **Model Training**: Fitting an **XGBoost Regressor** on the training set.
6.  **Evaluation**: Measuring performance using the R-Squared score.

## 📈 Model Performance

- **Model**: XGBoost Regressor
- **Training R² Score**: **0.876**
- **Testing R² Score**: **0.502**

_Observation: The significant gap between training and testing scores indicates overfitting. This suggests the model has learned the training data too well but struggles to generalize. Future work would involve hyperparameter tuning (e.g., regularization, tree depth) to improve generalization._

## 🧠 Key Concepts & Learnings

- **XGBoost Regression**: Applying gradient boosting for regression tasks.
- **Data Imputation**: Strategies for handling missing numerical and categorical data.
- **Label Encoding**: Converting categorical text data into model-readable numeric values.
- **Model Evaluation**: Understanding R-Squared and identifying overfitting.

## 💻 Setup & Usage

1.  Clone this repository.
2.  Install required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
3.  Open and run the `Big_Mart_Sales_Prediction.ipynb` notebook.
