#-----------------------
#1. Import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------
#2. Reading & Splitting 

dataset = pd.read_csv(r"C:\Users\G AKHILA\Desktop\Datasets\Machine Learning\Salary_Data.csv")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

#-----------------------
#3. x-Train, x-Test, y-Train & y-Test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.15, random_state=0)

#------------------------
#4. Building Simple Linear Regerssion model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#------------------------
#5. Predictions

y_pred = regressor.predict(x_test)

#------------------------
#6. Plot the graphs of training set

plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Test)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#------------------------
#7. Plot the graphs of testing set

plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
