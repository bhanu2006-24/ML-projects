# 🎗️ Breast Cancer Classification

**Type**: Binary Classification  
**Algorithm**: Logistic Regression  
**Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset (569 samples, 30 features)

## 📖 Project Description

This project implements a machine learning model to classify breast cancer tumors as **Malignant** (harmful) or **Benign** (non-harmful) based on features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. Early diagnosis of breast cancer can significantly improve survival rates, making automated classification systems a vital tool in medical diagnostics.

## 📊 Model Performance

The Logistic Regression model demonstrates strong predictive capability with high accuracy on both training and test data:

- **Training Accuracy**: **~94.95%**
- **Testing Accuracy**: **~92.98%**
- **Insight**: The model generalizes well to unseen data, with a small gap between training and test performance, indicating it is not overfitting.

## 🔑 Key Concepts Learned

- **Sklearn Datasets**: Efficiently loading and using built-in datasets like `load_breast_cancer`.
- **Data Analysis**:
  - Statistical summary using `describe()`.
  - Understanding class distribution (Benign vs Malignant).
  - Grouping data to analyze feature mean differences between classes.
- **Logistic Regression**: Using this algorithm for binary classification tasks in the medical domain.
- **Predictive System**: Building a system to take in raw medical feature data (30 numerical inputs) and output a clinical diagnosis.

## 🚀 Usage

### 💻 Running the Project

1. **Open on Google Colab**:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12BcpWDOkRRb1L9Mai43pervN641kqzDW?usp=sharing)
   The notebook is designed to run in Google Colab.

2. **Run Locally**:

   ```bash
   # Navigate to the directory
   cd "Breast Cancer Classification"

   # Launch Jupyter Notebook
   jupyter notebook Breast_Cancer_Classification.ipynb
   ```

## 🛠Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
