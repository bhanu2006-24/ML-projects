# 🪨 Rock vs Mine Prediction using SONAR Data

## 📋 Project Overview
This project uses Machine Learning to predict whether an object detected by SONAR is a **Rock** or a **Mine** (metal cylinder). The model is trained on SONAR data containing 60 features representing energy levels at different frequencies.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qROcP_vZHOwLl1QMRz22AAHLikPtg9CM?usp=sharing)

## 📊 Dataset
- **Source**: SONAR (Sound Navigation and Ranging) data
- **File**: `sonar.csv`
- **Total Samples**: 208 observations
- **Features**: 60 numerical features representing energy levels at different frequency bands
- **Target Variable**: Binary classification
  - `R` = Rock (111 samples)
  - `M` = Mine/Metal Cylinder (97 samples)
- **Feature Range**: Continuous values between 0.0 and 1.0
- **No Missing Values**: Clean dataset ready for modeling

## 🎯 Objective
Build a binary classification model to accurately distinguish between rocks and mines based on SONAR signal patterns.

## 🛠️ Technologies Used
- Python
- Pandas & NumPy (Data manipulation)
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (Visualization)
- Jupyter Notebook

## 🧠 Machine Learning Approach

### 1. Data Collection & Preprocessing
- Loaded SONAR dataset with 208 samples and 60 features
- Verified data integrity (no missing values)
- Analyzed statistical measures of all features
- Separated features (X) and labels (Y)

### 2. Exploratory Data Analysis
- Examined data distribution and class balance
- Statistical summary of all 60 frequency features
- Label distribution: Rocks vs Mines

### 3. Data Splitting
- **Training Set**: 80% (stratified split to maintain class distribution)
- **Test Set**: 20% (for unbiased evaluation)
- Random state used for reproducibility

### 4. Model Selection & Training
- **Algorithm**: Logistic Regression
- **Reasoning**: Effective for binary classification with numerical features
- Trained on 166 samples (80% of dataset)

### 5. Model Evaluation
- **Training Accuracy**: 83.42%
- **Testing Accuracy**: 76.19%
- Model shows good generalization with acceptable train-test gap

## 📈 Results

### Model Performance
| Metric | Training Data | Test Data |
|--------|--------------|------------|
| **Accuracy** | 83.42% | 76.19% |
| **Dataset Size** | 166 samples | 42 samples |

### Key Findings
- ✅ The Logistic Regression model successfully learned to differentiate between rocks and metal cylinders
- ✅ Training accuracy of **83.42%** demonstrates good pattern recognition
- ✅ Test accuracy of **76.19%** shows reasonable generalization to unseen data
- ✅ The ~7% difference between train and test accuracy indicates minimal overfitting
- ✅ Model performs well considering the small dataset size and high-dimensional feature space (60 features)

### Interpretation
The model can correctly identify whether a SONAR reading represents a rock or mine approximately **3 out of 4 times** on new, unseen data, which is valuable for underwater object detection applications.

## 🚀 How to Run
1. Clone this repository
2. Open the Jupyter notebook: `SONAR_Rock_vs_Mine_Prediction.ipynb`
3. Or use the **[Google Colab link](https://colab.research.google.com/drive/1qROcP_vZHOwLl1QMRz22AAHLikPtg9CM?usp=sharing)** to run directly in your browser
4. Run all cells sequentially

## 📚 Key Learnings

### Technical Skills
- **Binary Classification**: Implemented Logistic Regression for two-class problem
- **SONAR Signal Processing**: Understood how frequency-based features represent physical objects
- **High-Dimensional Data**: Worked with 60 features for only 208 samples
- **Model Evaluation**: Calculated and interpreted accuracy metrics for training and test sets
- **Train-Test Split**: Applied stratified splitting to maintain class distribution

### Machine Learning Concepts
- **Logistic Regression**: Linear classifier using sigmoid function for probability estimates
- **Overfitting Detection**: Monitored train-test gap to assess generalization
- **Feature Importance**: All 60 frequency bands contribute to classification
- **Real-World ML**: Dealt with small dataset constraints (208 samples)

### Domain Knowledge
- **SONAR Technology**: Learned how sound waves reflect differently off rocks vs metals
- **Signal Processing**: Understood frequency-based feature extraction
- **Military/Naval Applications**: Mine detection use case

## 📁 Project Structure
```
Rock Vs Mine Prediction/
│
├── SONAR_Rock_vs_Mine_Prediction.ipynb  # Main notebook
├── sonar.csv                             # Dataset
└── README.md                             # Project documentation
```

## 🔮 Future Improvements
- Experiment with different ML algorithms (SVM, Random Forest, Neural Networks)
- Hyperparameter tuning
- Feature importance analysis
- Deploy the model as a web application

---
**Author**: Bhanu Pratap Saini  
**Date**: December 2025
