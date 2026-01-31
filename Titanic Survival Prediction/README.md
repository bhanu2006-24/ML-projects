# Titanic Survival Prediction

This project builds a predictive model to determine which passengers survived the Titanic shipwreck. It involves data cleaning, exploratory data analysis, feature engineering, and training a Logistic Regression model.

## 🔗 Project Notebook

[View the Notebook](./Titanic_Survival_Prediction.ipynb) | [Open in Colab](https://colab.research.google.com/drive/1jua3O9aRuqWUQv59tfrzwdQ8SGB4jq6K?usp=sharing)

## 📊 Dataset

The analysis uses the classic **Titanic dataset** (loaded via Seaborn), containing **891** records.

- **Target**: `survived` (0 = No, 1 = Yes)
- **Key Features**:
  - **Demographics**: `sex`, `age`, `adult_male`, `who` (man, woman, child)
  - **Socioeconomic**: `pclass` (Passenger Class), `fare`, `class`
  - **Family**: `sibsp` (Siblings/Spouses), `parch` (Parents/Children), `alone`
  - **Travel**: `embarked`, `embark_town`

## 🛠 Technologies Used

- **Python**
- **Pandas & NumPy**: For data manipulation and scientific computing.
- **Seaborn & Matplotlib**: For data visualization (count plots, etc.).
- **Scikit-learn**:
  - `LabelEncoder`: For converting categorical features.
  - `train_test_split`: For splitting the data.
  - `LogisticRegression`: The classification algorithm.
  - `accuracy_score`: For model evaluation.

## 🚀 Project Workflow

1.  **Data Collection**: Loading the dataset from the Seaborn library.
2.  **Data Cleaning**:
    - Dropping the `deck` column due to excessive missing values (~77%).
    - Imputing missing `age` values with the mean.
    - Imputing missing `embarked` and `embark_town` values with the mode.
3.  **Exploratory Data Analysis (EDA)**: Visualizing survival rates by gender, class, and survival status.
4.  **Feature Encoding**:
    - Using **Label Encoding** for categorical features (`sex`, `embarked`, `class`, `who`, etc.).
    - Converting boolean columns (`adult_male`, `alone`) to integers.
5.  **Addressing Data Leakage**:
    - Identified and removed the `alive` column as it was a proxy for the target variable (resulting in 100% accuracy initially).
6.  **Model Training**: Training a **Logistic Regression** model.
7.  **Evaluation**: Measuring accuracy scores on training and test datasets.
8.  **Prediction System**: Testing the model on a hypothetical passenger.

## 📈 Model Performance

- **Training Accuracy**: **~83.43%**
- **Test Accuracy**: **~78.21%**

## 🧠 Key Concepts & Learnings

- **Data Leakage**: The importance of identifying features like `alive` that unintentionally reveal the target variable, leading to overly optimistic results.
- **Data Imputation**: Handling missing values using statistical measures (mean/mode).
- **Label Encoding**: Transforming categorical text data into numerical format for machine learning.
- **Logistic Regression**: A fundamental algorithm for binary classification tasks.

## 💻 Setup & Usage

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```
3.  Run the notebook `Titanic_Survival_Prediction.ipynb`.
