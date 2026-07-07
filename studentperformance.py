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
#plt.bar(avg_score.index,avg_score.values)
#plt.show()
#plt.hist(df["math score"])
#plt.show()

x = df[["reading score","writing score"]]

y = df ["math score"]

X_train, X_test, y_train, y_test = train_test_split( x,y,test_size =0.2,random_state= 42)

model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test,predictions)

print("Mean Absolute Error:", mae)

new_student = [[80,75]]

predictions= model.predict(new_student)
print("predicted math score:", predictions)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)