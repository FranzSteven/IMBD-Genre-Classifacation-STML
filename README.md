# IMBD-Genre-Classifacation-STML

This project focuses on classifying the genre of movies based on their titles and descriptions using machine learning. The dataset used is the "Genre Classification Dataset" from IMDB, available on Kaggle.

## Project Structure

- **genre_classification.ipynb:** Jupyter Notebook containing the code for data loading, preprocessing, model training, evaluation, and SHAP analysis.
- **README.md:** This file.

## Dataset

The dataset is sourced from Kaggle and contains the following files:

- **train_data.txt:** Training data with movie IDs, titles, genres, and descriptions.
- **test_data.txt:** Test data with movie IDs, titles, and descriptions.
- **test_data_solution.txt:** Ground truth genre labels for the test data.

## Dependencies

The following libraries are required:

- pandas
- numpy
- matplotlib
- scikit-learn
- nltk
- kagglehub
- shap

You can install them using:

!pip install pandas numpy matplotlib scikit-learn nltk kagglehub shap

## Usage

1. **Download the dataset:**
   - Download the "Genre Classification Dataset" from Kaggle: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
   - Extract the downloaded zip file to the `data/` directory.

2. **Run the Jupyter Notebook:**
   - Open the `genre_classification.ipynb` notebook.
   - Execute the cells sequentially to load the data, preprocess it, train the model, evaluate the performance, and visualize feature importance using SHAP.

## Model

The project uses a Linear Support Vector Classifier (LinearSVC) for genre classification. The text data is preprocessed by removing punctuation, tokenizing, removing stop words, and lemmatizing. TF-IDF vectorization is used to convert the text into numerical features.

## Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation is performed to ensure robustness.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) is used to understand the importance of features in the model's predictions. It provides insights into which words or phrases contribute most to the classification of different genres.

