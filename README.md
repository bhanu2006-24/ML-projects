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

**[View Project](./House%20Price%20Prediction/) | [Open in Colab](https://colab.research.google.com/drive/1RsJQe7wvWQdLk-YayUNwlqFLpT47FtPx?usp=sharing)**

---

### 4. 📰 [Fake News Prediction](./Fake%20News%20Prediction/)

**Type**: Binary Text Classification  
**Algorithm**: Logistic Regression with TF-IDF  
**Dataset**: Fake News Dataset (20,800 articles, 5 features)

**Description**: Classify news articles as Real or Fake using Natural Language Processing (NLP) and machine learning. The model analyzes textual content and author information to distinguish between credible news and misinformation, utilizing TF-IDF vectorization to extract meaningful features from text data.

**Model Performance**:

- Training Accuracy: **98.64%**
- Testing Accuracy: **97.91%**
- Excellent generalization with only 0.73% train-test gap

**Key Concepts Learned**:

- Natural Language Processing (NLP) and text preprocessing
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Text classification with Logistic Regression
- Working with sparse matrices and high-dimensional text data
- NLTK for stopword removal and Porter Stemmer for text normalization
- Feature engineering: combining author and title information
- Handling missing values in textual datasets
- Real-world application in combating misinformation

**[View Project](./Fake%20News%20Prediction/) | [Open in Colab](https://colab.research.google.com/drive/1pnSk5JCm-XpaWxvGs5GQ3AD7UWytj2Hg?usp=sharing)**

---

### 5. 🏦 [Loan Status Prediction](./Loan%20Status%20Prediction/)

**Type**: Binary Classification  
**Algorithm**: Support Vector Machine (SVM)  
**Dataset**: Loan Prediction Dataset (614 samples, 13 features)

**Description**: Automate the loan eligibility process by predicting whether a loan application should be approved or rejected. The model analyzes customer demographics, financial information, and credit history to classify applications, helping finance companies make real-time, data-driven decisions.

**Model Performance**:

- Training Accuracy: **79.86%**
- Testing Accuracy: **83.33%**
- Demonstrates robust generalization with consistent performance on unseen data

**Key Concepts Learned**:

- Handling missing data in both categorical and numerical columns
- Label Encoding for categorical feature transformation
- Support Vector Machine (SVM) implementation with linear kernel
- Analyzing the impact of Credit History on loan approval
- Building a predictive system for new loan applications
- Data visualization with Seaborn for feature relationships

**[View Project](./Loan%20Status%20Prediction/) | [Open in Colab](https://colab.research.google.com/drive/1PIuMJ73qPA-hwil63F2WWWVVj0MJZODS?usp=sharing)**

---

## 🎯 Overall Learning Journey

Through these machine learning projects, I have gained hands-on experience with real-world datasets and practical ML applications:

### 🔧 Technical Skills Developed

- **Data Preprocessing**: Cleaning, normalization with StandardScaler, feature scaling, and text preprocessing
- **Exploratory Data Analysis**: Statistical analysis, distribution visualization, and correlation studies
- **Natural Language Processing (NLP)**:
  - Text cleaning and normalization
  - TF-IDF vectorization for feature extraction
  - Stopword removal and stemming with NLTK
  - Working with sparse matrices
- **Algorithm Implementation**:
  - Logistic Regression for signal classification and text classification
  - Support Vector Machines (SVM) for medical prediction
  - XGBoost Regressor for price prediction
- **Model Evaluation**: Training vs test accuracy, R² score, MAE, generalization assessment, overfitting detection
- **Python Libraries**: Proficiency in NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, and NLTK

### 🧠 Machine Learning Concepts Mastered

- **Supervised Learning**: Binary classification, text classification, and regression techniques
- **Feature Engineering**: Working with high-dimensional (60 features), medical, geographic, and textual data
- **Text Processing**: TF-IDF vectorization, stemming, and stopword removal for NLP tasks
- **Train-Test Split**: Proper stratified data partitioning for model validation
- **Model Selection**: Choosing appropriate algorithms for specific problem types
  - Classification: Logistic Regression (signals & text), SVM
  - Regression: XGBoost for continuous predictions
- **Overfitting & Generalization**:
  - Rock vs Mine: 7% gap (acceptable for small dataset)
  - Diabetes: 1.4% gap (excellent generalization)
  - House Price: 11% gap (good performance with complex model)
  - Fake News: 0.73% gap (excellent generalization with text data)
- **Standardization**: Critical preprocessing for distance-based algorithms like SVM
- **Class Imbalance**: Handling imbalanced datasets in medical applications
- **Regression Metrics**: R² Score (coefficient of determination), Mean Absolute Error
- **Ensemble Methods**: Gradient boosting with XGBoost for improved predictions
- **Sparse Matrices**: Efficient handling of high-dimensional text features

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
- Classified fake news with **97.91% test accuracy** using NLP techniques
- Predicted loan eligibility with **83.33% test accuracy** using SVM
- Worked with diverse datasets: 208 samples (signal processing) to 20,800 articles (text classification)
- Handled feature spaces from 8 to 60 dimensions, plus high-dimensional text vectors
- Demonstrated understanding of model generalization and validation
- Mastered classification (binary and text), regression, and NLP problems

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
pip install numpy pandas scikit-learn matplotlib seaborn jupyter nltk xgboost
```

## 📞 Connect With Me

Feel free to explore the projects, provide feedback, or reach out for collaboration!

---

**Note**: Each project folder contains its own detailed README with specific information about the dataset, methodology, and results.

**Last Updated**: December 2025
