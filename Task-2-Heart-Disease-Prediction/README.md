# ❤️ Heart Disease Prediction using Machine Learning

This project performs **Exploratory Data Analysis (EDA)** and builds **machine learning models** to predict heart disease in patients using clinical features.

---

## 🎯 Objective
To analyze patient data and predict the likelihood of heart disease using supervised machine learning techniques.  

The project helps doctors or healthcare professionals **identify high-risk patients early**.

---

## 📊 Dataset Overview
Dataset: `heart_disease.csv`  

**Common Columns include:**  
- age, sex, chest pain type (cp), resting blood pressure (trestbps)  
- cholesterol (chol), fasting blood sugar (fbs), resting ECG (restecg)  
- max heart rate (thalach), exercise-induced angina (exang)  
- ST depression (oldpeak), slope, number of major vessels (ca)  
- thal, target (presence of heart disease)  

---

## 🧩 Steps Performed

### 🧹 1. Data Preprocessing
- Handled missing values  
- Encoded categorical variables  
- Scaled numerical features (StandardScaler / MinMaxScaler)  

### 📊 2. Exploratory Data Analysis (EDA)
- Visualized feature distributions  
- Studied correlations between features and target  
- Checked outliers using boxplots and Z-score  

### 🤖 3. Model Training
Implemented and compared: 
- Random Forest Classifier  
  

### 📈 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score, AUC  
- Confusion Matrix and ROC Curve  
- Selected the best performing model and saved it  

### 💾 5. Model Saving
```python
import joblib
joblib.dump(best_model, 'models/heart_model.pkl')
