# 🎓 Student Math Score Prediction using Machine Learning

A Machine Learning regression project that predicts a student's **Math Score** using academic performance and demographic information from the Students Performance dataset.

This project demonstrates the complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, model comparison, visualization, and model persistence.

---

## 📌 Project Overview

The objective of this project is to predict a student's **math score** based on features such as:

* Reading Score
* Writing Score
* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch Type
* Test Preparation Course

Multiple regression models were trained and compared to determine which performed best.

---

## 📂 Dataset

**Dataset:** Students Performance Dataset

Features:

* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch
* Test Preparation Course
* Reading Score
* Writing Score

Target Variable:

* Math Score

---

## 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

---

## 📊 Exploratory Data Analysis (EDA)

The project includes:

* Dataset inspection
* Summary statistics
* Missing value analysis
* Correlation analysis
* Group-wise performance analysis
* Histograms
* Bar charts

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Train-Test Split
* One-Hot Encoding using `pd.get_dummies()`
* Feature Selection

---

## 🤖 Machine Learning Models

### 1. Linear Regression

Implemented Linear Regression to establish a baseline regression model.

Evaluation Metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Additional Analysis:

* Actual vs Predicted comparison
* Prediction error analysis
* Scatter plot visualization
* Model coefficient interpretation

---

### 2. Decision Tree Regressor

Implemented Decision Tree Regression and compared its performance with Linear Regression.

---

### 3. Random Forest Regressor

Implemented Random Forest Regression for improved predictive performance.

Additional Analysis:

* Feature Importance
* Model Comparison

---

## 📈 Model Evaluation

Models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Performance comparison was performed across all trained models.

---

## 📊 Visualizations

The project includes:

* Average Score Bar Chart
* Math Score Distribution
* Actual vs Predicted Scatter Plot
* Feature Importance Plot
* Model Performance Comparison

---

## 💾 Model Persistence

The final trained model is saved using Joblib.

```python
joblib.dump(model, "student_math_predictor.pkl")
```

---

## 📁 Project Structure

```
Student-Performance-Analysis/
│
├── studentsperformance.csv
├── studentperformance.py
├── prediction_results.csv
├── feature_importance.csv
├── student_math_predictor.pkl
├── README.md
```

---

## 📚 Machine Learning Concepts Covered

* Exploratory Data Analysis
* Feature Engineering
* One-Hot Encoding
* Train-Test Split
* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Regression Metrics
* Feature Importance
* Model Comparison
* Model Saving

---

## 🎯 Future Improvements

* Cross Validation
* Hyperparameter Tuning (GridSearchCV)
* Pipeline Implementation
* Streamlit Web Application
* Flask API Deployment
* Docker Support
* CI/CD Integration

---

## 👨‍💻 Author

**Rohit Jha**

AI & Machine Learning Student

GitHub: https://github.com/madebyrohitjha

---

⭐ If you found this project useful, consider giving it a star.
