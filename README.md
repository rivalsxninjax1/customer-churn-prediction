# Customer Churn Prediction

This repository contains a machine learning project that predicts customer churn using a Random Forest Classifier. The project uses grid search with cross-validation to optimize the model’s hyperparameters and includes evaluation metrics such as classification reports, confusion matrix, and ROC AUC score. It also displays feature importances to help understand which factors are most influential in predicting customer churn.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Pipeline](#model-and-pipeline)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer churn is a major concern for businesses, and predicting it helps companies identify customers who are likely to leave and take proactive measures. In this project, we build a churn prediction model using customer data that contains features such as demographic information, service usage, billing details, and contract type. The model is built with Python and uses the following libraries:

- **Pandas & NumPy:** Data manipulation
- **Scikit-Learn:** Machine learning (data splitting, model training, grid search, evaluation)
- **Matplotlib & Seaborn:** Data visualization

The model uses a Random Forest Classifier with a preprocessing pipeline that scales the data and imputes missing values.

## Dataset

The dataset is a CSV file (`data/customer_churn.csv`) with the following columns (among others):

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn` (Target variable: "Yes" or "No")

The file should be placed in a folder named `data` within the project directory.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/rivalsxninjax1/customer-churn-prediction.git
   cd customer-churn-prediction
# customer-churn-prediction
