# 📰 Fake News Prediction

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pnSk5JCm-XpaWxvGs5GQ3AD7UWytj2Hg?usp=sharing)

## 📌 Project Overview

A machine learning project that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) and Logistic Regression. This project demonstrates the application of text preprocessing, feature extraction using TF-IDF, and binary classification to combat misinformation.

## 🎯 Objective

Build a reliable classifier to identify fake news articles by analyzing their textual content and author information, helping users distinguish between credible and unreliable news sources.

## 📊 Dataset

- **Source**: Kaggle - Fake News Prediction Dataset
- **Size**: 20,800 news articles
- **Features**: 5 columns
  - `id`: Unique identifier for each article
  - `title`: The headline of the news article
  - `author`: Author of the article
  - `text`: Full text content of the article (may be incomplete)
  - `label`: Binary classification
    - `0`: Real News
    - `1`: Fake News

### Data Characteristics
- **Missing Values**: 
  - Title: 558 missing values
  - Author: 1,957 missing values
  - Text: 39 missing values
- **Data Distribution**: Balanced dataset with both real and fake news samples

## 🔧 Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Replaced null values with empty strings
- **Feature Engineering**: Combined `author` and `title` into a single `content` feature
- **Text Processing**:
  - Removed stopwords (common English words like 'the', 'is', 'and')
  - Applied stemming using Porter Stemmer to reduce words to root form
  - Cleaned special characters and normalized text

### 2. Feature Extraction
- **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency)
  - Converts text data into numerical feature vectors
  - Captures word importance relative to the entire corpus
  - Enables machine learning algorithms to process textual data

### 3. Model Training
- **Algorithm**: Logistic Regression
- **Train-Test Split**: 80-20 ratio (stratified)
- **Libraries Used**:
  - `NLTK`: Natural language processing and stopwords
  - `Scikit-learn`: Machine learning implementation
  - `Pandas` & `NumPy`: Data manipulation
  - `re`: Regular expressions for text cleaning

## 📈 Model Performance

### Accuracy Metrics
| Dataset | Accuracy |
|---------|----------|
| **Training Data** | **98.64%** |
| **Test Data** | **97.91%** |

### Key Observations
- ✅ **Excellent Performance**: 97.91% test accuracy demonstrates strong classification capability
- ✅ **Minimal Overfitting**: Only 0.73% gap between training and test accuracy
- ✅ **Great Generalization**: Model performs consistently on unseen data
- ✅ **Reliable Predictions**: High confidence in real vs fake classification

## 💡 Key Learnings

### Technical Skills
- **Natural Language Processing (NLP)**:
  - Text preprocessing and cleaning techniques
  - Stopword removal and stemming
  - Feature extraction from unstructured text data
  
- **Text Vectorization**:
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Converting textual data to numerical representations
  - Understanding word importance in document context

- **Machine Learning**:
  - Logistic Regression for binary text classification
  - Working with high-dimensional sparse matrices
  - Model evaluation for NLP tasks

### Domain Knowledge
- **Fake News Characteristics**: Understanding linguistic patterns in misinformation
- **Content Analysis**: Importance of author credibility and title analysis
- **Data Quality**: Handling missing values in real-world text datasets

### Best Practices
- Feature combination (author + title) for improved predictions
- Stratified train-test split for balanced evaluation
- Text normalization for better model generalization
- Comprehensive preprocessing pipeline

## 🚀 How to Use

### Prerequisites
```bash
pip install numpy pandas scikit-learn nltk kagglehub
```

### Download NLTK Stopwords
```python
import nltk
nltk.download('stopwords')
```

### Running the Project
1. Download the dataset using KaggleHub
2. Run the Jupyter notebook for step-by-step execution
3. Train the model and evaluate performance
4. Test predictions on new articles

### Making Predictions
```python
# Example prediction
new_article = "Author Name Article Title Here"
# Preprocess and vectorize
prediction = model.predict(vectorizer.transform([new_article]))
if prediction[0] == 0:
    print("The news is Real")
else:
    print("The news is Fake")
```

## 🔍 Model Details

- **Input**: Combined text (author + title)
- **Preprocessing**: Stemming, stopword removal, lowercasing
- **Feature Vector**: TF-IDF matrix
- **Classifier**: Logistic Regression
- **Output**: Binary classification (0 = Real, 1 = Fake)

## 📚 Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **NLP** | NLTK (Porter Stemmer, Stopwords) |
| **ML Algorithm** | Scikit-learn (Logistic Regression) |
| **Feature Extraction** | TF-IDF Vectorizer |
| **Platform** | Google Colab, Jupyter Notebook |
| **Dataset** | Kaggle (KaggleHub) |

## 🎓 Real-World Applications

- **News Verification**: Automated fact-checking systems
- **Social Media Filtering**: Identifying misinformation on platforms
- **Browser Extensions**: Real-time fake news alerts
- **Journalistic Tools**: Assisting fact-checkers and reporters
- **Educational Platforms**: Teaching media literacy

## 📊 Future Improvements

- [ ] Experiment with deep learning models (LSTM, BERT)
- [ ] Incorporate full article text for better context
- [ ] Add multi-class classification (satire, opinion, factual)
- [ ] Implement ensemble methods for improved accuracy
- [ ] Deploy as a web application or API
- [ ] Fine-tune hyperparameters for optimization
- [ ] Add visualization of important words/features

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to fork the repository and submit pull requests.

---

**Project Status**: ✅ Completed  
**Last Updated**: December 2025
