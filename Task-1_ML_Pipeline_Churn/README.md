# Task 2: End-to-End ML Pipeline for Customer Churn Prediction

## Project Objective
The goal of this project is to **build a reusable and production-ready machine learning pipeline** for predicting customer churn using the **Telco Customer Churn Dataset**.  
The pipeline is implemented with **Scikit-learn's Pipeline API** and includes preprocessing, model training, hyperparameter tuning, and pipeline export.

---

## Dataset
- **Name:** Telco Customer Churn Dataset  
- **Description:** Customer information including demographics, account info, and service usage, with a target column `Churn` indicating whether the customer left (`Yes`) or stayed (`No`).  
- **Target Column:** `Churn` (converted to 0 = No, 1 = Yes)  

---

## Project Instructions Implemented
1. **Data Preprocessing with Pipeline**
   - Scaling of numerical features
   - One-hot encoding of categorical features
   - Handling missing values using imputation  

2. **Model Training**
   - Logistic Regression
   - Random Forest Classifier  

3. **Hyperparameter Tuning**
   - Used `GridSearchCV` to find the best parameters for both models  

4. **Pipeline Export**
   - Exported the complete pipeline using `joblib` for production use  

---

## Skills Gained
- Constructing **ML pipelines** using Scikit-learn  
- Performing **hyperparameter tuning** with GridSearchCV  
- Exporting models and pipelines for **reusability**  
- Applying **production-ready ML practices**

---

## Installation
Install the required Python packages:

```bash
pip install pandas numpy scikit-learn joblib
