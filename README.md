# 🤖 Machine Learning Projects Portfolio

Welcome to my Machine Learning learning journey! This repository contains hands-on ML projects covering different aspects of machine learning, from fundamentals to advanced techniques.

---

## 📚 Projects

### 1. 🩨 [Rock vs Mine Prediction](./Rock%20Vs%20Mine%20Prediction/)
**Type**: Binary Classification  
**Algorithm**: Logistic Regression  
**Dataset**: SONAR Signals (208 samples, 60 frequency-based features)

**Description**: Predict whether a SONAR signal represents a rock or a mine (metal cylinder) using machine learning classification. The project analyzes frequency patterns in SONAR data to distinguish between two different underwater objects.

**Model Performance**:
- Training Accuracy: **83.42%**
- Testing Accuracy: **76.19%**
- Minimal overfitting with ~7% train-test gap

**Key Concepts Learned**:
- Binary classification with Logistic Regression
- SONAR signal processing and frequency analysis
- High-dimensional data handling (60 features for 208 samples)
- Train-test split with stratified sampling
- Model evaluation and overfitting detection
- Real-world application in underwater object detection

**[View Project](./Rock%20Vs%20Mine%20Prediction/) | [Open in Colab](https://colab.research.google.com/drive/1qROcP_vZHOwLl1QMRz22AAHLikPtg9CM?usp=sharing)**

---

### 2. 💉 [Diabetes Prediction](./Diabetes%20Prediction/)
**Type**: Binary Classification  
**Algorithm**: Support Vector Machine (SVM)  
**Dataset**: PIMA Indians Diabetes (768 patients, 8 medical features)

**Description**: Predict whether a person has diabetes based on various health indicators like glucose levels, blood pressure, BMI, and other medical measurements. Uses the famous PIMA Indians Diabetes dataset from the National Institute of Diabetes and Digestive and Kidney Diseases.

**Model Performance**:
- Training Accuracy: **78.66%**
- Testing Accuracy: **77.27%**
- Excellent generalization with only 1.4% train-test gap

**Key Concepts Learned**:
- Medical data preprocessing and StandardScaler normalization
- Support Vector Machines (SVM) with linear kernel
- Working with healthcare datasets and ethical considerations
- Feature correlation analysis for health indicators
- Handling class imbalance (65% non-diabetic vs 35% diabetic)
- Clinical interpretation of ML models
- Stratified sampling for medical data

**[View Project](./Diabetes%20Prediction/) | [Open in Colab](https://colab.research.google.com/drive/1yq4BrIMRgKL-5doRALDwTe_v4q8L-9EL?usp=sharing)**

---

### 3. 🏠 [House Price Prediction](./House%20Price%20Prediction/)
**Type**: Regression  
**Algorithm**: XGBoost Regressor  
**Dataset**: California Housing (20,640 samples, 8 features)

**Description**: Predict median house prices for California districts using demographic and geographic features. Uses the California Housing dataset from the 1990 U.S. Census, incorporating factors like median income, house age, average rooms, population, and location coordinates.

**Model Performance**:
- Training R² Score: **94.19%**
- Test R² Score: **82.87%**
- Mean Absolute Error (Test): **0.313** (≈$31,300)

**Key Concepts Learned**:
- Regression modeling with XGBoost gradient boosting
- Working with geographic data (latitude/longitude)
- R² Score and Mean Absolute Error evaluation metrics
- Real estate pricing factors and census data analysis
- Feature engineering with demographic variables
- Model generalization with 11% train-test gap
- Ensemble learning methods and boosting algorithms

**[View Project](./House%20Price%20Prediction/)**

---

## 🎯 Overall Learning Journey

Through these machine learning projects, I have gained hands-on experience with real-world datasets and practical ML applications:

### 🔧 Technical Skills Developed
- **Data Preprocessing**: Cleaning, normalization with StandardScaler, and feature scaling
- **Exploratory Data Analysis**: Statistical analysis, distribution visualization, and correlation studies
- **Algorithm Implementation**: 
  - Logistic Regression for signal classification
  - Support Vector Machines (SVM) for medical prediction
  - XGBoost Regressor for price prediction
- **Model Evaluation**: Training vs test accuracy, R² score, MAE, generalization assessment, overfitting detection
- **Python Libraries**: Proficiency in NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, and Seaborn

### 🧠 Machine Learning Concepts Mastered
- **Supervised Learning**: Binary classification and regression techniques
- **Feature Engineering**: Working with high-dimensional (60 features), medical, and geographic data
- **Train-Test Split**: Proper stratified data partitioning for model validation
- **Model Selection**: Choosing appropriate algorithms for specific problem types
  - Classification: Logistic Regression, SVM
  - Regression: XGBoost for continuous predictions
- **Overfitting & Generalization**: 
  - Rock vs Mine: 7% gap (acceptable for small dataset)
  - Diabetes: 1.4% gap (excellent generalization)
  - House Price: 11% gap (good performance with complex model)
- **Standardization**: Critical preprocessing for distance-based algorithms like SVM
- **Class Imbalance**: Handling imbalanced datasets in medical applications
- **Regression Metrics**: R² Score (coefficient of determination), Mean Absolute Error
- **Ensemble Methods**: Gradient boosting with XGBoost for improved predictions

### 💡 Problem-Solving Approach
- Breaking down complex problems from data loading to model deployment
- Choosing appropriate algorithms based on data characteristics:
  - Logistic Regression for linearly separable frequency data
  - SVM for complex medical decision boundaries
- Iterative improvement through experimentation and metric analysis
- Comprehensive documentation and reproducibility practices

### 🚀 Tools & Platforms
- **Jupyter Notebooks**: Interactive development and analysis
- **Google Colab**: Cloud-based experimentation with easy sharing
- **Git & GitHub**: Version control and project portfolio management
- **Scikit-learn**: Industry-standard ML library

### 📈 Practical Achievements
- Successfully classified SONAR signals with **76.19% test accuracy**
- Predicted diabetes with **77.27% test accuracy** on medical data
- Predicted house prices with **82.87% R² score** on regression task
- Worked with diverse datasets: 208 samples (signal processing) to 20,640 census blocks (housing)
- Handled feature spaces from 8 to 60 dimensions
- Demonstrated understanding of model generalization and validation
- Mastered both classification and regression problems

## 📈 Future Goals
- Explore deep learning with TensorFlow/PyTorch
- Work on NLP and Computer Vision projects
- Implement end-to-end ML pipelines
- Deploy models as web applications
- Participate in Kaggle competitions

## 🛠️ Setup & Installation

To run these projects locally:

```bash
# Clone the repository
git clone <repository-url>

# Navigate to project directory
cd "ML projects"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install common dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## 📞 Connect With Me
Feel free to explore the projects, provide feedback, or reach out for collaboration!

---

**Note**: Each project folder contains its own detailed README with specific information about the dataset, methodology, and results.

**Last Updated**: December 2025
