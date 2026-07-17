import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("studentsperformance.csv")

#profile = ProfileReport(df)
#profile.to_file("eda_report.html")
#print(df.head())
#print(df.info())
#print(df[["math score","reading score","writing score"]].describe())
#print(df.isnull().sum())
#print(df["reading score"].max())
#print(df["math score"].min())
#print(df["math score"].mean())
#print(df.groupby("gender")[["math score","reading score","writing score"]].mean())

#print(df.groupby("test preparation course")
#      [["reading score","writing score","math score"]].mean())

#print(df[["math score","reading score","writing score"]].corr(numeric_only=True))

#import matplotlib.pyplot as plt
#avg_score = df[["math score","reading score","writing score"]].mean()
#plt.bar(avg_score.index, avg_score.values)
#plt.show()

#plt.hist(df["math score"])
#plt.show()

x = df[["reading score", "writing score"]]
y = df["math score"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# Predict a new student's math score
new_student = pd.DataFrame({
    "reading score": [80],
    "writing score": [75]
})

new_prediction = model.predict(new_student)

print("Predicted math score:", round(new_prediction[0], 2))

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print(comparison.head(20))


comparison = pd.DataFrame({
    "Actual Math Score": y_test.values,
    "Predicted Math Score": predictions
})

print(comparison.head(20))

comparison["Error"] = comparison["Actual Math Score"] - comparison["Predicted Math Score"]

print(comparison.head(20))

comparison.to_csv("prediction_results.csv", index=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted Math Scores")

plt.show()

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

print(coefficients)

X = df[["reading score", "writing score"]]

print(df.columns)

X = df.drop("math score", axis=1)
y = df["math score"]
X = pd.get_dummies(X, drop_first=True)

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))


linear_mae = mean_absolute_error(y_test, predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, predictions))
linear_r2 = r2_score(y_test, predictions)

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)  
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE :", mae)
print("RMSE:", rmse)
print("R² :", r2)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE :", mae)
print("RMSE:", rmse)
print("R² :", r2)


decision_tree_mae = mae
decision_tree_rmse = rmse
decision_tree_r2 = r2

results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "MAE": [linear_mae, decision_tree_mae],
    "RMSE": [linear_rmse, decision_tree_rmse],
    "R²": [linear_r2, decision_tree_r2]
})

print(results)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

random_forest_mae = mean_absolute_error(y_test, predictions)
random_forest_rmse = np.sqrt(mean_squared_error(y_test, predictions))
random_forest_r2 = r2_score(y_test, predictions)

print("Random Forest")
print("MAE :", random_forest_mae)
print("RMSE:", random_forest_rmse)
print("R² :", random_forest_r2)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        decision_tree_mae,
        random_forest_mae
    ],
    "RMSE": [
        linear_rmse,
        decision_tree_rmse,
        random_forest_rmse
    ],
    "R²": [
        linear_r2,
        decision_tree_r2,
        random_forest_r2
    ]
})

print(results)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(results["Model"], results["MAE"])

plt.title("Model Comparison (MAE)")
plt.xlabel("Model")
plt.ylabel("MAE")

plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.show()

importance.to_csv(
    "feature_importance.csv",
    index=False
)