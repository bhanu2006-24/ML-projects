# 🏠 House Price Prediction using XGBoost

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RsJQe7wvWQdLk-YayUNwlqFLpT47FtPx?usp=sharing)

## 📋 Project Overview
This project uses Machine Learning to predict house prices based on various features like median income, house age, average rooms, location, and other demographic factors. The model is trained on the California Housing dataset using XGBoost, a powerful gradient boosting algorithm.

## 📊 Dataset

### California Housing Dataset
- **Source**: Scikit-learn built-in dataset (derived from 1990 U.S. Census)
- **Total Samples**: 20,640 census block groups
- **Features**: 8 numerical predictive attributes
  1. **MedInc**: Median income in block group
  2. **HouseAge**: Median house age in block group
  3. **AveRooms**: Average number of rooms per household
  4. **AveBedrms**: Average number of bedrooms per household
  5. **Population**: Block group population
  6. **AveOccup**: Average number of household members
  7. **Latitude**: Block group latitude
  8. **Longitude**: Block group longitude
- **Target Variable**: Median house value for California districts (in hundreds of thousands of dollars)
- **No Missing Values**: Clean dataset ready for modeling

## 🎯 Objective
Build a regression model to accurately predict house prices based on demographic and geographic features.

## 🛠️ Technologies Used
- Python
- Pandas & NumPy (Data manipulation)
- XGBoost (Machine Learning)
- Matplotlib & Seaborn (Visualization)
- Scikit-learn (Metrics and utilities)
- Jupyter Notebook / Google Colab

## 🧠 Machine Learning Approach

### 1. Data Collection & Preprocessing
- Loaded California Housing dataset (20,640 samples)
- Verified data integrity (no missing values)
- Analyzed statistical measures of all features
- Created DataFrame with feature columns and target

### 2. Exploratory Data Analysis
- Examined data distribution and feature relationships
- Statistical summary of all 8 features
- Analyzed correlation between features and house prices

### 3. Data Splitting
- **Training Set**: 80% (for model training)
- **Test Set**: 20% (for unbiased evaluation)
- Random state used for reproducibility

### 4. Model Selection & Training
- **Algorithm**: XGBoost Regressor
- **Reasoning**: 
  - Highly effective for regression tasks
  - Handles complex non-linear relationships
  - Resistant to overfitting with proper tuning
  - Excellent performance on tabular data

### 5. Model Evaluation
- **Training R² Score**: 94.19%
- **Test R² Score**: 82.87%
- **Mean Absolute Error (Test)**: 0.313 (in hundreds of thousands)
- Good generalization with ~11% train-test gap

## 📈 Results

### Model Performance
| Metric | Training Data | Test Data |
|--------|--------------|------------|
| **R² Score** | 94.19% | 82.87% |
| **Mean Absolute Error** | 0.197 | 0.313 |

### Key Findings
- ✅ The XGBoost model achieved excellent performance with **82.87% R² score** on test data
- ✅ Training R² of **94.19%** shows the model learned the patterns effectively
- ✅ Mean Absolute Error of **0.313** means predictions are typically within $31,300 of actual prices
- ✅ The model captures **82.87%** of the variance in house prices
- ✅ Good balance between model complexity and generalization

### Interpretation
The model effectively predicts California house prices with high accuracy, explaining over 82% of the variance in the data. The difference between training and test scores suggests some overfitting, but the test performance remains strong and useful for real-world applications.

## 🚀 How to Run
1. Clone this repository
2. Open the Jupyter notebook: `House_Price_Prediction.ipynb`
3. Install required dependencies: `pip install numpy pandas matplotlib seaborn xgboost scikit-learn`
4. Run all cells sequentially

## 📚 Key Learnings

### Technical Skills
- **Regression Modeling**: Implemented XGBoost for continuous value prediction
- **Gradient Boosting**: Understanding ensemble methods and boosting algorithms
- **Feature Analysis**: Working with geographic and demographic data
- **Model Evaluation**: Using R² score and MAE for regression performance
- **Real-world Dataset**: Handling census data with realistic constraints

### Machine Learning Concepts
- **XGBoost Algorithm**: Gradient boosting framework for high performance
- **R² Score (Coefficient of Determination)**: Measuring proportion of variance explained
- **Mean Absolute Error**: Understanding prediction accuracy in original units
- **Overfitting Monitoring**: Balancing training and test performance
- **Regression vs Classification**: Different problem types and metrics

### Domain Knowledge
- **Real Estate Pricing**: Understanding factors that influence house prices
  - Location (latitude/longitude) as strong predictors
  - Income as a key economic indicator
  - Property characteristics (rooms, age)
  - Population density effects
- **Census Data**: Working with demographic block group data
- **California Housing Market**: Context of 1990s housing prices

## 📁 Project Structure
```
House Price Prediction/
│
├── House_Price_Prediction.ipynb  # Main notebook with analysis and model
└── README.md                      # Project documentation
```

## 🔮 Future Improvements
- Feature engineering (e.g., rooms per capita, income per capita)
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Try other algorithms (Random Forest, LightGBM, Neural Networks)
- Feature importance analysis to identify key price drivers
- Deploy the model as a web application for real-time predictions
- Include cross-validation for more robust evaluation
- Add more recent housing data for temporal comparison

---
**Author**: Bhanu Pratap Saini  
**Date**: December 2025
