# Calories Burnt Prediction

This project builds a machine learning regression model to predict calorie expenditure during exercise based on physiological and activity data.

## 🔗 Project Notebook

[View the Notebook](./Calories_Burnt_Prediction.ipynb) | [Open in Colab](https://colab.research.google.com/drive/10nJ4V_X6cDqtLKvapAEXOvI4oJ6Hr_Gw?usp=sharing)

## 📊 Dataset

The dataset combines two files from Kaggle (`calories.csv` and `exercise.csv`), totalling **15,000** records.

- **Target Variable**: `Calories` (burned)
- **Features**:
  - **Demographics**: `Gender`, `Age`
  - **Physiological**: `Height`, `Weight`, `Body_Temp`
  - **Activity**: `Duration` (minutes), `Heart_Rate` (bpm)

## 🛠 Technologies Used

- **Python**
- **Pandas & NumPy**: For data manipulation and analysis.
- **Seaborn & Matplotlib**: For data visualization (correlation heatmaps, distribution plots).
- **XGBoost**: `XGBRegressor` for high-performance gradient boosting.
- **Scikit-learn**:
  - `train_test_split`: To split the data.
  - `mean_absolute_error`: For model evaluation.

## 🚀 Project Workflow

1.  **Data Collection**: Loading and merging the exercise and calories datasets.
2.  **Data Preprocessing**:
    - Combined datasets using concatenation potentially assuming row-alignment or effectively joining on User_ID context.
    - Key feature engineering: Converting categorical 'Gender' to numerical values (male=0, female=1).
    - Dropping `User_ID` as it is non-predictive.
3.  **Exploratory Data Analysis**: Analyzing relationships between duration, heart rate, and calories burnt.
4.  **Model Training**: Training an **XGBoost Regressor** to handle non-linear relationships and interactions.
5.  **Evaluation**: Measuring performance using Mean Absolute Error (MAE).

## 📈 Model Performance

- **Mean Absolute Error (MAE)**: **~1.48**
  - This indicates that, on average, the model's prediction is within ~1.5 calories of the actual value, showing excellent predictive capability for this dataset.

## 🧠 Key Concepts & Learnings

- **Gradient Boosting**: Using XGBoost for regression tasks on structured data.
- **Feature Correlation**: Understanding that `Duration` and `Heart_Rate` are highly correlated with calorie burn.
- **Data Integration**: Merging multiple data sources to create a complete feature set.

## 💻 Setup & Usage

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy seaborn matplotlib xgboost scikit-learn
    ```
3.  Run the notebook `Calories_Burnt_Prediction.ipynb`.
