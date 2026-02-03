# 📧 Spam Mail Prediction

**Type**: Binary Classification  
**Algorithm**: Logistic Regression  
**Dataset**: SMS Spam Collection (5572 messages)

## 📖 Project Description

This project builds a machine learning model to classify SMS messages as either "Spam" or "Ham" (legitimate). With the rise of unsolicited marketing and scam messages, automated spam detection is crucial for user experience and security. This system uses Natural Language Processing (NLP) techniques to analyze the text content of messages and predict their category.

## 📊 Model Performance

The Logistic Regression model achieved excellent results, demonstrating high accuracy and robustness:

- **Training Accuracy**: **96.77%**
- **Testing Accuracy**: **96.68%**
- The minimal gap between training and testing accuracy indicates that the model is well-generalized and not overfitting.

## 🔑 Key Concepts Learned

- **Text Preprocessing**:
  - Handling null values in the dataset.
  - Label Encoding: Converting 'spam' to 0 and 'ham' to 1.
- **Feature Extraction**:
  - **TF-IDF Vectorization**: Converting raw text data into numerical feature vectors that represent the importance of words in the messages.
- **Logistic Regression**:
  - Applying Logistic Regression for binary text classification.
- **Model Evaluation**:
  - Using Accuracy Score to validate the model's performance on unseen data.
- **Predictive System**:
  - Creating a pipeline to take a raw text message, transform it using the trained vectorizer, and predict whether it is spam or ham.

## 🚀 Usage

### 📂 Dataset

The dataset consists of 5,572 SMS messages tagged as either `ham` (legitimate) or `spam`.

- **ham**: Legitimate messages (e.g., "See you later", "Ok lar... Joking wif u oni...")
- **spam**: Unsolicited messages (e.g., "Free entry in 2 a wkly comp...", "WINNER!! As a valued network customer...")

### 💻 Running the Project

1. **Open on Google Colab**:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MWB9H0gGTRg4pbUKn13acplT002Z58p3?usp=sharing)
   The notebook is designed to run in Google Colab, where it can automatically download the dataset.
2. **Run Locally**:

   ```bash
   # Navigate to the directory
   cd "Spam Mail Prediction"

   # Launch Jupyter Notebook
   jupyter notebook Spam_Mail_Prediction.ipynb
   ```

## 🛠Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
