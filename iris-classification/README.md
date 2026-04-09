# Iris Flower Classification

This project focuses on classifying iris flowers into different species using basic machine learning models. It is part of a supervised learning practice series and demonstrates how different algorithms perform on the same dataset.

## Overview

The Iris dataset contains measurements of flower features such as:
- Sepal length
- Sepal width
- Petal length
- Petal width

Using these features, the task is to predict the species of the flower.

## Models Used

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naive Bayes

## Approach

The workflow followed in this project:

- Loaded the dataset using pandas  
- Checked for missing values  
- Split the data into training and testing sets  
- Applied feature scaling using Standardization  
- Built models using pipelines  
- Used GridSearchCV to tune hyperparameters (KNN and Logistic Regression)  
- Evaluated models using Accuracy, Precision, and Recall  

## Results

All three models performed very well on the dataset and achieved perfect accuracy.

## Model Comparison

| Model                | Accuracy | Precision | Recall |
|---------------------|----------|----------|--------|
| KNN                 | 1.0      | 1.0      | 1.0    |
| Logistic Regression | 1.0      | 1.0      | 1.0    |
| Naive Bayes         | 1.0      | 1.0      | 1.0    |

## Conclusion

All models achieved perfect performance on the Iris dataset.

K-Nearest Neighbors (KNN) is considered the best model in this project as it performed optimally after hyperparameter tuning and works well for this type of dataset.

KNN is simple, effective, and particularly suitable for datasets where classes are clearly separable, as seen in the Iris dataset.

## Project Structure

iris-classification/
│
├── iris_flower.ipynb      # Jupyter Notebook with full implementation  
├── Iris.csv               # Dataset  
├── requirements.txt       # Dependencies  
└── README.md              # Project documentation  

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Open Jupyter Notebook:
   jupyter notebook

3. Run the notebook:
   - Open iris_flower.ipynb  
   - Run all cells  

## Notes

This project is created for learning and practice purposes as part of a collection of supervised machine learning models.
