# 🎬 Movie Recommendation System

**Type**: Content-Based Recommendation System  
**Algorithm**: Cosine Similarity with TF-IDF Vectorization  
**Dataset**: TMDB 5000 Movie Dataset (approx. 4803 movies)

## 📖 Project Description

This project builds a specialized recommendation system that suggests movies to users based on their favorite films. By analyzing the content of movies—such as genres, keywords, cast, and directors—the system identifies and recommends movies with similar characteristics. This "Content-Based Filtering" approach ensures that recommendations align closely with the user's specific tastes.

## 📊 Methodology & Performance

- **Content-Based Filtering**: Unlike collaborative filtering which relies on user interactions, this system focuses on the attributes of the items themselves.
- **Feature Engineering**: Combines critical metadata (`genres`, `keywords`, `tagline`, `cast`, `director`) to create a comprehensive "content profile" for each movie.
- **Vectorization**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text descriptions into numerical vectors, emphasizing unique and important terms.
- **Similarity Calculation**: employs **Cosine Similarity** to measure the angle between movie vectors, effectively ranking how close two movies are in the feature space.

## 🔑 Key Concepts Learned

- **Recommendation Systems**: Understanding the difference between content-based and collaborative filtering.
- **Text Data Processing**:
  - Handling missing values in text features.
  - Concatenating multiple text columns to form a unified feature set.
- **TF-IDF Vectorization**: Transforming unstructured text data into meaningful numerical representations.
- **Cosine Similarity**: Mathematically determining the similarity between documents (movies).
- **Fuzzy String Matching**: Using `difflib` to handle user input errors and find the closest matching movie title in the database.

## 🚀 Usage

### 💻 Running the Project

1. **Open on Google Colab**:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/130K2T1qDxHJVRHBo60GB-mOyyPs0awLB?usp=sharing)
   The notebook is designed to run in Google Colab, where it automatically handles dataset downloading.

2. **Run Locally**:

   ```bash
   # Navigate to the directory
   cd "Movie Recommendation System"

   # Launch Jupyter Notebook
   jupyter notebook Movie_Recommendation_System.ipynb
   ```

   _Note: Ensure you have the `movies.csv` file available or allow the script to download it._

## 🛠Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Difflib
