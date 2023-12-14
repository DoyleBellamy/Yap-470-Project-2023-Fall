# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

# Importing the dataset
dosya_yolu = '/home/umutozdemir/Desktop/Yap-470-Project-2023-Fall/DataCreation/output.xlsx'
dataset = pd.read_excel(dosya_yolu)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# Applying K-Means clustering on X_train
kmeans = KMeans(n_clusters=2, random_state=0)  # Assuming two clusters
y_kmeans_train = kmeans.fit_predict(X_train)

# Calculate cluster centers
cluster_centers = kmeans.cluster_centers_

# Create a Linear Regression model for each cluster
models = []
for i in range(len(cluster_centers)):
    cluster_mask = (y_kmeans_train == i)
    X_cluster = X_train[cluster_mask]
    y_cluster = y_train[cluster_mask]
    
    model = LinearRegression()
    model.fit(X_cluster, y_cluster)
    models.append(model)

# Predicting the cluster for each data point in X_test
y_kmeans_test = kmeans.predict(X_test)

# Predicting the y values for each data point in X_test using the corresponding model
y_pred_test = np.zeros_like(y_kmeans_test, dtype=float)
for i, model in enumerate(models):
    cluster_mask = (y_kmeans_test == i)
    X_test_cluster = X_test[cluster_mask]
    y_pred_test[cluster_mask] = model.predict(X_test_cluster)

# Convert predictions to binary (0 or 1)
threshold = 0.5
y_pred_test_binary = (y_pred_test > threshold).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Visualising the Test set results
plt.scatter(X_test[:, 0], y_test, color='red', label='Gerçek Değerler')
plt.scatter(X_test[:, 0], y_pred_test_binary, color='blue', label='Binary Tahmin Değerleri')
plt.axhline(y=threshold, color='black', linestyle='--', label=f'Eşik Değeri = {threshold}')
plt.title('K-Means Clustering + Linear Regression - Binary Predictions')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
