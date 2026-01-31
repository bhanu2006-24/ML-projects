# Credit Card Fraud Detection

This project predicts whether a credit card transaction is fraudulent or legitimate using a Logistic Regression model. It addresses the critical challenge of class imbalance in fraud detection datasets.

## 🔗 Project Notebook

[View the Colab Notebook](https://colab.research.google.com/drive/1V3PtEejy2SlC80p4HdTLn3sI6YxAL5kL?usp=sharing)

## 📊 Dataset

The dataset used contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days.

- **Features**: `V1`, `V2`, ... `V28` (Principal component obtained with PCA), `Time`, and `Amount`.
- **Target**: `Class` (0 = Normal Transaction, 1 = Fraudulent Transaction).

**Important Note**: The original dataset is highly unbalanced, with 492 frauds out of 284,807 transactions.

## 🛠 Technologies Used

- **Python**
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For model building, training, evaluation, and data splitting.

## 🚀 Project Workflow

1.  **Data Collection & Processing**:
    - Loading the dataset and inspecting its structure.
    - Checking for missing values and duplicates.
    - Analyzing the distribution of the target variable (`Class`).
2.  **Handling Imbalance (Undersampling)**:
    - Separating the data into Legitimate and Fraudulent transactions.
    - Creating a sample dataset containing all 492 fraud transactions and a random sample of 492 normal transactions to create a balanced dataset.
3.  **Data Splitting**:
    - Splitting the balanced dataset into Features (X) and Target (Y).
    - Splitting the data into training and testing sets (80% Train, 20% Test) using stratified sampling.
4.  **Model Training**:
    - Using the **Logistic Regression** algorithm.
    - Training the model on the training data.
5.  **Model Evaluation**:
    - Predicting outcomes on both training and test data.
    - Evaluating performance using the **Accuracy Score** metric.

## 📈 Model Performance

- **Model**: Logistic Regression
- **Training Accuracy**: ~94.41%
- **Test Accuracy**: ~90.36%

The model performs well on the balanced dataset, demonstrating the effectiveness of undersampling in handling highly skewed data distributions.

## 🧠 Key Concepts & Learnings

- **Class Imbalance**: Understanding the challenges of datasets where one class significantly outnumbers the other.
- **Undersampling**: A technique to balance the dataset by reducing the number of samples in the majority class.
- **Binary Classification**: Classifying transactions into two distinct categories.
- **Logistic Regression**: Applying this algorithm for binary classification tasks.

## 💻 Setup & Usage

1.  Clone this repository or download the project files.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  Run the notebook `Credit_Card_Fraud_Detection.ipynb` to see the full analysis and prediction pipeline.
