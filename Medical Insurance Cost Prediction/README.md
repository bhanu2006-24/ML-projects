# Medical Insurance Cost Prediction

This project builds a machine learning model to predict individual medical insurance costs based on demographic and lifestyle information. It utilizes Linear Regression to estimate charges, providing insights into how different factors contribute to healthcare expenses.

## 🔗 Project Notebook

[View the Colab Notebook](https://colab.research.google.com/drive/1uK58D38QdKzVnzv40D_opWd5QwbTkxJ8?usp=sharing)

## 📊 Dataset

The dataset consists of medical insurance records with the following attributes:

- **Features**: `age`, `sex`, `bmi` (Body Mass Index), `children` (number of children/dependents), `smoker` (smoking status), `region` (residential area).
- **Target**: `charges` (individual medical costs billed by health insurance).
- **Size**: 1338 entries.

### Key Data Insights:

- **Age Distribution**: Broad range of ages represented.
- **BMI**: Distributed around a mean of ~30.
- **Categorical Columns**: `sex` (male/female), `smoker` (yes/no), `region` (southeast/southwest/northeast/northwest).

## 🛠 Technologies Used

- **Python**
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Matplotlib & Seaborn**: For data visualization (count plots, distance plots).
- **Scikit-learn**: For model building, splitting data, and evaluation metrics.

## 🚀 Project Workflow

1.  **Data Collection & Inspection**: Loading the dataset and performing initial checks for missing values and data types.
2.  **Exploratory Data Analysis (EDA)**: Visualizing the distribution of numerical features (Age, BMI, Children, Charges) and categorical features (Sex, Smoker, Region).
3.  **Data Preprocessing (Encoding)**:
    - **Sex**: male → 0, female → 1
    - **Smoker**: yes → 0, no → 1
    - **Region**: southeast → 0, southwest → 1, northeast → 2, northwest → 3
4.  **Data Splitting**: Separating data into features (X) and target (Y) and splitting into training and testing sets (80% Train, 20% Test).
5.  **Model Training**: Using the **Linear Regression** algorithm to learn the relationship between features and insurance charges.
6.  **Model Evaluation**: using the R² (R-squared) score to measure model performance.
7.  **Predictive System**: Building a function to input new data and get a cost prediction.

## 📈 Model Performance

- **Model**: Linear Regression
- **Training R² Score**: ~0.7515
- **Test R² Score**: ~0.7447

The model achieves consistent performance on both training and test sets, explaining approximately 75% of the variance in medical insurance costs.

## 🧠 Key Concepts & Learnings

- **Encoding Categorical Data**: converting text labels (`sex`, `smoker`, `region`) into numerical values suitable for regression.
- **Linear Regression**: Understanding how linear relationships can model cost predictions.
- **R-squared Metric**: Evaluating regression models by measuring goodness of fit.
- **Feature Importance**: Seeing potentially high impact features like smoking status on cost variance.

## 💻 Setup & Usage

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Run the notebook `Medical_Insurance_Cost_Prediction.ipynb`.
