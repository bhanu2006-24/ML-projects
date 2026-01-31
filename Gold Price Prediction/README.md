# Gold Price Prediction

This project focuses on predicting the price of Gold (GLD) based on several financial indicators using a Random Forest Regressor. It demonstrates the application of machine learning in financial forecasting.

## 🔗 Project Notebook

[View the Colab Notebook](https://colab.research.google.com/drive/1vnv22Lb_Dy3Gn6s42rSR1xKx7klsLsRY?usp=sharing)

## 📊 Dataset

The dataset used for this project is the **Gold Price Data**, which includes historical data on:

- **SPX**: S&P 500 Index
- **GLD**: Gold Price
- **USO**: United States Oil Fund
- **SLV**: Silver Price
- **EUR/USD**: Euro to US Dollar Exchange Rate

The goal is to predict the **GLD** price using the other financial indicators.

## 🛠 Technologies Used

- **Python**
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization and correlation analysis.
- **Scikit-learn**: For model building, training, and evaluation.

## 🚀 Project Workflow

1.  **Data Collection & Processing**:
    - Loading the dataset and inspecting its structure.
    - Checking for missing values and understanding statistical measures.
2.  **Exploratory Data Analysis (EDA)**:
    - Analyzing correlations between different financial assets using heatmaps.
    - Visualizing the distribution of the target variable (GreenGLD Price).
3.  **Data Preparation**:
    - Separating features (`SPX`, `USO`, `SLV`, `EUR/USD`) and target (`GLD`).
    - Splitting the data into training and testing sets (80% Train, 20% Test).
4.  **Model Training**:
    - Using the **Random Forest Regressor** algorithm.
    - Training the model on the training dataset.
5.  **Model Evaluation**:
    - Predicting Gold prices on the test set.
    - Evaluating performance using the **R-squared error** metric.
    - Visualizing the comparison between Actual vs. Predicted prices.

## 📈 Model Performance

- **Model**: Random Forest Regressor
- **R-squared Error**: ~0.989

The Random Forest Regressor achieved an exceptionally high R-squared score, indicating that the model explains nearly 99% of the variance in the Gold price data.

## 🧠 Key Concepts & Learnings

- **Random Forest Regression**: Understanding ensemble learning methods for regression tasks.
- **Correlation Analysis**: Identifying positive and negative correlations between gold and other assets (e.g., strong positive correlation with Silver).
- **Financial Forecasting**: Applying ML techniques to predict financial market trends.

## 💻 Setup & Usage

1.  Clone this repository or download the project files.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Run the notebook `Gold_Price_Prediction.ipynb` to see the full analysis and prediction pipeline.
