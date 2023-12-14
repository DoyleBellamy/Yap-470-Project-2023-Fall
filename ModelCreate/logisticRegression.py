# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Importing the dataset
dosya_yolu = '/home/umutozdemir/Desktop/Yap-470-Project-2023-Fall/DataCreation/output.xlsx'
dataset = pd.read_excel(dosya_yolu)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Training the Logistic Regression model on the Training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Visualising the Test set results
plt.scatter(X_test[:, 0], y_test, color='red', label='Gerçek Değerler')  # Gerçek değerleri kırmızı renkte göster
plt.scatter(X_test[:, 0], y_pred, color='blue', label='Tahmin Değerleri')  # Tahmin değerlerini mavi renkte göster
plt.title('Salary vs Experience (Test set) - Logistic Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Evaluate the Logistic Regression model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
