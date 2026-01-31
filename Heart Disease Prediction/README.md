# Heart Disease Prediction

This project focuses on predicting whether a person has heart disease based on various medical attributes using a Logistic Regression model. It demonstrates the application of machine learning in healthcare for early disease detection.

## 🔗 Project Notebook

[View the Colab Notebook](https://colab.research.google.com/drive/1Pc5BhWTdmWbuYiChNNryV1OXxdp-jez6?usp=sharing)

## 📊 Dataset

The dataset used in this project contains medical information about patients. It includes 303 samples and the following features:

- **age**: Age of the patient
- **sex**: Gender of the patient
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure
- **chol**: Serum cholestoral in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by flourosopy
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
- **target**: 1 = Defective Heart (Disease), 0 = Healthy Heart

## 🛠 Technologies Used

- **Python**
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For model building, training, and evaluation.

## 🚀 Project Workflow

1.  **Data Collection & Processing**:
    - Loading the dataset into a Pandas DataFrame.
    - Inspecting the data (missing values, statistical measures).
    - Checking the distribution of the target variable.
2.  **Data Splitting**:
    - Separating features (X) and target (Y).
    - Splitting the data into training and testing sets (80% Train, 20% Test) using stratified sampling.
3.  **Model Training**:
    - Using the **Logistic Regression** algorithm, which is suitable for binary classification tasks.
    - Training the model on the training data.
4.  **Model Evaluation**:
    - Predicting outcomes on both training and test data.
    - Evaluating performance using the **Accuracy Score** metric.
5.  **Predictive System**:
    - Building a system to predict heart disease for new input data.

## 📈 Model Performance

- **Model**: Logistic Regression
- **Training Accuracy**: ~85.12%
- **Test Accuracy**: ~81.97%

The model shows consistent performance on both training and test sets, indicating good generalization without significant overfitting.

## 🧠 Key Concepts & Learnings

- **Binary Classification**: Applying Logistic Regression to classify patients into two categories (Healthy vs. Defective Heart).
- **Medical Data Analysis**: Understanding and working with medical features like blood pressure, cholesterol, and heart rate.
- **Stratified Sampling**: Ensuring the training and test sets have a similar proportion of class labels.
- **Model Evaluation**: Using accuracy score to assess the effectiveness of the classification model.

## 💻 Setup & Usage

1.  Clone this repository or download the project files.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  Run the notebook `Heart_Disease_Prediction.ipynb` to see the full analysis and prediction pipeline.
