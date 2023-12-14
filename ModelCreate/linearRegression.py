# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dosya_yolu = '/home/umutozdemir/Desktop/Yap-470-Project-2023-Fall/DataCreation/output.xlsx'
dataset = pd.read_excel(dosya_yolu)
X = dataset.iloc[:, :-1].values
print(X.shape)
y = dataset.iloc[:, -1].values
print(y.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)


plt.scatter(X_train[:, 0], y_train, color='red')  # Assuming you want to visualize the first feature
plt.plot(X_train[:, 0], regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



# Visualising the Test set results
plt.scatter(X_test[:, 0], y_test, color='red')  # Assuming you want to visualize the first feature
plt.plot(X_test[:, 0], regressor.predict(X_test), color='blue')  # Use X_test here
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test[:, 0], y_test, color='red', label='Gerçek Değerler')  # Gerçek değerleri kırmızı renkte göster
plt.scatter(X_test[:, 0], regressor.predict(X_test), color='blue', label='Tahmin Değerleri')  # Tahmin değerlerini mavi renkte göster
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

