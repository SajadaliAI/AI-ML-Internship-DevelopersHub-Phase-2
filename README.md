# 📊 Customer Churn Prediction using Machine Learning

An end-to-end machine learning project for predicting whether a customer is likely to **churn (leave a service)** using the Telco Customer Churn dataset.

The project demonstrates a reusable Scikit-learn pipeline with data preprocessing, model training, hyperparameter tuning, evaluation, and model export.

---

## 🎯 Project Objective

The goal is to build a machine learning system that predicts customer churn based on customer demographics, services, account information, and usage-related features.

The target variable is:

* `0` → Customer did not churn
* `1` → Customer churned

---

## 📂 Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer information such as:

* Demographics
* Account information
* Services
* Contract details
* Payment methods
* Tenure
* Monthly charges
* Total charges

**Target Column:** `Churn`

---

## 🔄 Machine Learning Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Numerical & Categorical Preprocessing
     ↓
Logistic Regression / Random Forest
     ↓
GridSearchCV
     ↓
Model Evaluation
     ↓
Best Model Pipeline
     ↓
Joblib Export
```

---

## 🧹 Data Preprocessing

The project uses Scikit-learn's `Pipeline` and `ColumnTransformer`.

### Numerical Features

* Missing values → Median imputation
* Feature scaling → `StandardScaler`

### Categorical Features

* Missing values → Most-frequent imputation
* Encoding → `OneHotEncoder`

The `customerID` column is removed because it is an identifier rather than a useful predictive feature.

---

## 🤖 Models

Two classification algorithms are trained and evaluated:

### Logistic Regression

Used as a simple and interpretable classification model.

### Random Forest Classifier

Used as a tree-based ensemble model capable of learning nonlinear relationships.

---

## 🔧 Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation is used to search for suitable model hyperparameters.

Parameters tuned include:

### Logistic Regression

* `C`
* `solver`

### Random Forest

* `n_estimators`
* `max_depth`
* `min_samples_split`

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

The best Random Forest pipeline is exported after evaluation.

---

## 💾 Model Export

The complete trained pipeline is saved using Joblib:

```python
joblib.dump(rf_grid.best_estimator_, "customer_churn_pipeline.pkl")
```

Because the complete pipeline is saved, preprocessing and prediction steps can be reused together.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Machine Learning
* Classification
* Feature Preprocessing
* Hyperparameter Tuning

---

## 📦 Installation

Install the required packages:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## ▶️ Run

Make sure the dataset is available in the project directory:

```text
TelcoCustomerChurn.csv
```

Then run the Python script:

```bash
python your_script.py
```

---

## 📁 Output

After training, the project generates:

```text
customer_churn_pipeline.pkl
```

This file contains the trained Random Forest pipeline and can be reused for future predictions.

---

## 📚 Skills Demonstrated

* End-to-end machine learning workflow
* Scikit-learn Pipelines
* ColumnTransformer
* Data preprocessing
* Categorical encoding
* Feature scaling
* Missing-value imputation
* Logistic Regression
* Random Forest
* GridSearchCV
* Cross-validation
* Model evaluation
* Model serialization with Joblib
