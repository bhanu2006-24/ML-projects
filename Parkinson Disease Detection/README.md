# Parkinson's Disease Detection

This project involves building a Support Vector Machine (SVM) model to detect Parkinson's disease based on biomedical voice measurements. The model analyzes various vocal features to classify individuals as either healthy or having Parkinson's disease.

## 🔗 Project Notebook

[View the Notebook](./Parkinson_Disease_Detection.ipynb) | [Open in Colab](https://colab.research.google.com/drive/1HkixxZ1-ArMGomk2bp4uny0xK_Ayf49H?usp=sharing)

## 📊 Dataset

The dataset consists of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD).

- **Size**: 195 rows, 24 columns.
- **Target Variable**: `status` (1 = Parkinson's, 0 = Healthy).
- **Key Features**:
  - **Frequency parameters**: `MDVP:Fo(Hz)` (Average vocal fundamental frequency), `MDVP:Fhi(Hz)` (Maximum), `MDVP:Flo(Hz)` (Minimum).
  - **Amplitude & Frequency variations**: `MDVP:Jitter(%)`, `MDVP:Shimmer`, etc.
  - **Tonal components**: `NHR` (Noise-to-Harmonics Ratio), `HNR`.
  - **Non-linear dynamics**: `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`.

## 🛠 Technologies Used

- **Python**
- **Pandas & NumPy**: For data manipulation and processing.
- **Scikit-learn**:
  - `train_test_split`: To split the data.
  - `StandardScaler`: For feature standardization.
  - `svm.SVC`: Support Vector Classifier for model training.
  - `accuracy_score`: For evaluation.

## 🚀 Project Workflow

1.  **Data Loading & Analysis**: Loading the CSV and checking for missing values (none found) and statistical distribution.
2.  **Data Preprocessing**:
    - Separating features ($X$) and target ($Y$).
    - Splitting data into training (80%) and testing (20%) sets.
    - **Standardization**: Using `StandardScaler` to transform the data so that it has a mean of 0 and variance of 1, which is crucial for SVM performance.
3.  **Model Training**: Training a **Support Vector Machine (SVM)** model with a linear kernel.
4.  **Evaluation**: Calculating accuracy on both training and test sets.
5.  **Predictive System**: Building a function to predict disease status for new input data.

## 📈 Model Performance

- **Training Accuracy**: ~88.5%
- **Test Accuracy**: ~87.2%

## 🧠 Key Concepts & Learnings

- **Support Vector Machine (SVM)**: A powerful supervised learning algorithm effective for classification in high-dimensional spaces.
- **Data Standardization**: The importance of scaling features when using distance-based algorithms like SVM to prevent features with larger magnitudes from dominating.
- **Feature Importance**: Understanding how vocal features like Jitter, Shimmer, and fundamental frequency variations correlate with Parkinson's disease.

## 💻 Setup & Usage

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  Run the notebook `Parkinson_Disease_Detection.ipynb`.
