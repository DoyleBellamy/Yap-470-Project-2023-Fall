# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dosya_yolu = '/home/umutozdemir/Desktop/Yap-470-Project-2023-Fall/DataCreation/output.xlsx'
dataset = pd.read_excel(dosya_yolu)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=15)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying XGBoost on X_train
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_scaled, y_train)

# Predicting the y values for each data point in X_test
y_pred_xgb = xgb_classifier.predict(X_test_scaled)

# Convert predictions to binary (0 or 1)
threshold = 0.5
y_pred_xgb_binary = (y_pred_xgb > threshold).astype(int)

# Calculate confusion matrix for XGBoost predictions
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb_binary)
print("Confusion Matrix for XGBoost Predictions:")
print(conf_matrix_xgb)

# Visualising the Test set results for XGBoost
plt.scatter(X_test[:, 0], y_test, color='red', label='Gerçek Değerler')
plt.scatter(X_test[:, 0], y_pred_xgb_binary, color='green', label='XGBoost Binary Tahmin Değerleri')
plt.axhline(y=threshold, color='black', linestyle='--', label=f'Eşik Değeri = {threshold}')
plt.title('XGBoost - Binary Predictions')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
