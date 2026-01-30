# 🏦 Loan Status Prediction

**Type**: Binary Classification  
**Algorithm**: Support Vector Machine (SVM)  
**Dataset**: Loan Prediction Dataset (614 samples, 13 features)

## 📖 Project Description

This project implements a machine learning system to automate the loan eligibility process for a housing finance company. By analyzing various customer details—such as gender, marital status, education, number of dependents, and income—the model predicts whether a loan application should be approved or rejected.

The problem is a classic binary classification task where the goal is to predict the `Loan_Status` (Yes/No) based on historical data.

## 📊 Model Performance

The Support Vector Machine (SVM) model with a linear kernel achieved the following results:

- **Training Accuracy**: **79.86%**
- **Testing Accuracy**: **83.33%**
- The model demonstrates strong generalization capabilities, performing slightly better on unseen test data than on the training set.

## 🔑 Key Concepts Learned

- **Data Preprocessing**:
  - Handling missing values (imputation) in both categorical and numerical features.
  - Label Encoding to convert categorical variables (e.g., Gender, Married, Property_Area) into numerical format.
- **Exploratory Data Analysis (EDA)**:
  - Visualizing relationships between features like Education/Marital Status and Loan Status.
- **Support Vector Machines (SVM)**:
  - Implementing SVM with a linear kernel for binary classification.
  - Understanding the importance of the margin in separating classes.
- **Model Evaluation**:
  - Using Accuracy Score to evaluate performance on training and test datasets.
- **Predictive System**:
  - Building a system to take new input data and output a loan decision (Eligible/Not Eligible).

## 🚀 Usage

### 📂 Dataset

The dataset includes the following key features:

- **Gender, Married, Dependents, Education, Self_Employed**: Demographic info
- **ApplicantIncome, CoapplicantIncome**: Financial info
- **LoanAmount, Loan_Amount_Term**: Loan details
- **Credit_History**: Past credit behavior (Critical feature)
- **Property_Area**: Urban/Semiurban/Rural

### 💻 Running the Project

1. **Open on Google Colab**:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PIuMJ73qPA-hwil63F2WWWVVj0MJZODS?usp=sharing)
2. **Run Locally**:

   ```bash
   # Navigate to the directory
   cd "Loan Status Prediction"

   # Launch Jupyter Notebook
   jupyter notebook Loan_Status_Prediction_.ipynb
   ```

## 🛠Dependencies

- Python 3.x
- Pandas
- NumPy
- Seaborn
- Scikit-learn
