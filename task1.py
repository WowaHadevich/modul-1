from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_shuffled)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_shuffled, test_size=0.3, random_state=52)

print("Навчальна вибірка:", X_train.shape, y_train.shape)
print("Тестова вибірка:", X_test.shape, y_test.shape)

k_values = [1, 2, 3, 5, 9]

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    
    knn_classifier.fit(X_train, y_train)
    
    y_pred = knn_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("K =", k, "Точність класифікації:", accuracy)

best_accuracy = 0
best_k = 0

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("Найкраща величина K:", best_k)
print("Точність класифікації для найкращої величини K:", best_accuracy)
