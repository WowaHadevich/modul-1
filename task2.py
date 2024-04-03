import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

lower_bound = 0
upper_bound = 100
random_data = np.random.randint(lower_bound, upper_bound, size=1000)

normalized_data = (random_data - np.mean(random_data)) / np.std(random_data)

X_train, X_test = train_test_split(normalized_data, test_size=0.3, random_state=52)

best_k = None
best_mse = float('inf')
mse_values = []
for k in range(1, 11):
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train.reshape(-1, 1), X_train)  
    predictions = knn_regressor.predict(X_test.reshape(-1, 1))  
    mse = mean_squared_error(X_test, predictions) 
    mse_values.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_k = k

print(f"Найкраще значення K: {best_k} з MSE = {best_mse}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), mse_values, marker='o', linestyle='-', color='b')
plt.title('Залежність MSE від значення K')
plt.xlabel('Значення K')
plt.ylabel('MSE')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
