# Car Price Prediction

This project focuses on predicting the selling price of used cars based on various features such as the car's age, present price, kilometers driven, fuel type, seller type, transmission, and number of owners. It utilizes regression algorithms to build a predictive model.

## 🔗 Project Notebook

[**View Colab Notebook**](https://colab.research.google.com/drive/1vnv22Lb_Dy3Gn6s42rSR1xKx7klsLsRY?usp=sharing)

## 📊 Dataset

The dataset used in this project is the **Vehicle Dataset from CarDekho** (Kaggle).

- **Source:** [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- **Features:**
  - `Year`: Year of manufacture
  - `Present_Price`: Current ex-showroom price
  - `Kms_Driven`: Distance driven in kilometers
  - `Fuel_Type`: Petrol, Diesel, or CNG
  - `Seller_Type`: Dealer or Individual
  - `Transmission`: Manual or Automatic
  - `Owner`: Number of previous owners
- **Target:** `Selling_Price`

## 🛠️ Technologies Used

- **Python**
- **Pandas** & **NumPy** (Data Manipulation)
- **Matplotlib** & **Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning)

## ⚙️ Project Workflow

1.  **Data Collection:** Downloading the dataset using `kagglehub`.
2.  **Data Preprocessing:**
    - Checking for missing values.
    - **Categorical Encoding:** Converting categorical variables (`Fuel_Type`, `Seller_Type`, `Transmission`) into numerical values.
3.  **Train-Test Split:**
    - Splitting the data into training and testing sets (90% Train, 10% Test).
4.  **Model Training:**
    - **Linear Regression**: Standard linear modeling.
    - **Lasso Regression**: Linear regression with L1 regularization.
5.  **Model Evaluation:**
    - Evaluating models using **R² Score**.
    - Visualizing Actual Prices vs. Predicted Prices.

## 📈 Model Performance

| Model                 | Training R² Score | Test R² Score |
| :-------------------- | :---------------- | :------------ |
| **Linear Regression** | ~0.88             | ~0.84         |
| **Lasso Regression**  | ~0.84             | ~0.87         |

**Key Findings:**

- **Lasso Regression** performed better on the test set, indicating better generalization compared to the standard Linear Regression model.
- `Present_Price` and `Year` are significant predictors of the selling price.

## 🚀 How to Run

1.  Open the provided [Colab Notebook](https://colab.research.google.com/drive/1vnv22Lb_Dy3Gn6s42rSR1xKx7klsLsRY?usp=sharing).
2.  Run the cells sequentially to download the data, process it, and train the models.
