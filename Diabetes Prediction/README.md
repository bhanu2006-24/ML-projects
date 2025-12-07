# 💉 Diabetes Prediction using Machine Learning

## 📋 Project Overview
This project uses Machine Learning to predict whether a person has diabetes based on various diagnostic measurements. The model is trained on medical data containing features like glucose levels, blood pressure, BMI, and other health indicators.

## 🔗 Google Colab Notebook
**[Open in Google Colab](https://colab.research.google.com/drive/1yq4BrIMRgKL-5doRALDwTe_v4q8L-9EL?usp=sharing)**

## 📊 Dataset

### PIMA Indians Diabetes Dataset
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases
- **File**: `diabetes.csv`
- **Total Samples**: 768 patient records (all females, Pima Indian heritage)
- **Features**: 8 medical diagnostic measurements
  1. **Pregnancies**: Number of pregnancies
  2. **Glucose**: Plasma glucose concentration (2 hours in oral glucose tolerance test)
  3. **BloodPressure**: Diastolic blood pressure (mm Hg)
  4. **SkinThickness**: Triceps skin fold thickness (mm)
  5. **Insulin**: 2-Hour serum insulin (mu U/ml)
  6. **BMI**: Body mass index (weight in kg/(height in m)²)
  7. **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic influence)
  8. **Age**: Age in years
- **Target Variable**: 
  - `0` = Non-Diabetic (500 patients, 65.1%)
  - `1` = Diabetic (268 patients, 34.9%)
- **Class Imbalance**: Dataset shows imbalance toward non-diabetic cases

## 🎯 Objective
Build a binary classification model to accurately predict diabetes diagnosis based on medical diagnostic measurements.

## 🛠️ Technologies Used
- Python
- Pandas & NumPy (Data manipulation)
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (Visualization)
- Jupyter Notebook / Google Colab

## 🧠 Machine Learning Approach

### 1. Data Collection & Preprocessing
- Loaded PIMA Indians Diabetes dataset (768 patient records)
- Analyzed 8 medical diagnostic features
- Identified feature ranges and distributions
- Addressed zero values in certain features (potential missing data indicators)

### 2. Exploratory Data Analysis
- **Statistical Summary**:
  - Mean glucose level: 120.9 mg/dL
  - Mean BMI: 32.0 (indicating overweight population)
  - Mean age: 33.2 years
  - Mean outcome: 34.9% diabetes prevalence
- Examined correlation between features and diabetes outcome
- Analyzed distribution of diabetic vs non-diabetic patients

### 3. Data Standardization
- Applied **StandardScaler** to normalize feature values
- Ensures all features contribute equally to model training
- Critical for distance-based algorithms like SVM

### 4. Data Splitting
- **Training Set**: ~80% (stratified to maintain class distribution)
- **Test Set**: ~20% (for unbiased evaluation)
- Stratified split to preserve class imbalance ratio

### 5. Model Selection & Training
- **Algorithm**: Support Vector Machine (SVM) with linear kernel
- **Reasoning**: SVM effective for:
  - Binary classification tasks
  - High-dimensional medical data
  - Finding optimal decision boundary
- Model trained on standardized training data

### 6. Model Evaluation
- **Training Accuracy**: 78.66%
- **Testing Accuracy**: 77.27%
- Excellent generalization with minimal overfitting (<1.4% gap)

## 📈 Results

### Model Performance
| Metric | Training Data | Test Data |
|--------|--------------|------------|
| **Accuracy** | 78.66% | 77.27% |
| **Algorithm** | SVM (Linear Kernel) | - |
| **Dataset Size** | ~614 patients | ~154 patients |

### Key Findings
- ✅ **Strong Generalization**: Test accuracy (77.27%) very close to training accuracy (78.66%)
- ✅ **Minimal Overfitting**: Only 1.4% difference indicates excellent model robustness
- ✅ **Clinical Relevance**: ~77% accuracy for diabetes prediction is valuable for early screening
- ✅ **Balanced Performance**: Model works well despite class imbalance (65% vs 35%)
- ✅ **Feature Engineering**: StandardScaler preprocessing significantly improved SVM performance

### Clinical Interpretation
The model correctly predicts diabetes status in approximately **3 out of 4 patients** based on their medical measurements. This can serve as:
- A preliminary screening tool
- Risk assessment for diabetes development
- Support system for healthcare professionals (not a replacement)

### Important Medical Context
- Dataset focuses on Pima Indian females, a population with high diabetes prevalence
- Model performance may vary for other demographics
- Should be validated with diverse populations before clinical deployment

## 🚀 How to Run
1. Clone this repository
2. Open the Jupyter notebook: `Diabeties_Prediction.ipynb`
3. Or use the **[Google Colab link](https://colab.research.google.com/drive/1yq4BrIMRgKL-5doRALDwTe_v4q8L-9EL?usp=sharing)** to run directly in your browser
4. Run all cells sequentially

## 📚 Key Learnings

### Technical Skills
- **Support Vector Machines (SVM)**: Implemented linear kernel SVM for medical classification
- **Feature Standardization**: Applied StandardScaler for data normalization
- **Medical Data Analysis**: Worked with real-world healthcare dataset
- **Class Imbalance Handling**: Managed 65-35 split between classes
- **Model Evaluation**: Computed training and test accuracies to assess performance

### Machine Learning Concepts
- **SVM Classification**: Understanding margin maximization and support vectors
- **Kernel Methods**: Applied linear kernel for decision boundary
- **Feature Scaling Impact**: Learned importance of standardization for SVM
- **Generalization**: Achieved minimal overfitting (1.4% train-test gap)
- **Stratified Sampling**: Preserved class distribution in train-test split

### Healthcare & Domain Knowledge
- **Diabetes Risk Factors**: Understanding key health indicators
  - Glucose levels as primary indicator
  - BMI and its correlation with diabetes
  - Genetic factors (Pedigree Function)
  - Age and pregnancy history influence
- **PIMA Indians Study**: Context of high-risk population research
- **Medical Ethics**: Importance of model interpretability in healthcare
- **Screening vs Diagnosis**: Model as screening tool, not diagnostic device

## 📁 Project Structure
```
Diabetes Prediction/
│
├── Diabeties_Prediction.ipynb  # Main notebook with analysis and model
├── diabetes.csv                 # Dataset with medical measurements
└── README.md                    # Project documentation
```

## 🔮 Future Improvements
- Experiment with ensemble methods (XGBoost, LightGBM)
- Hyperparameter tuning using GridSearch or RandomSearch
- Feature importance analysis to identify key health indicators
- SHAP values for model interpretability
- Deploy the model as a web application for real-time predictions
- Include additional health metrics for improved accuracy

## ⚕️ Medical Disclaimer
This project is for educational purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---
**Author**: Bhanu Pratap Saini  
**Date**: December 2025
