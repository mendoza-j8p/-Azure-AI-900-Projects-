'''Un caso práctico sencillo sobre regresión lineal, utilizando Python y la biblioteca scikit-learn. 
Vamos a predecir las ventas de helados en función de la temperatura.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Datos: Temperatura (x) y Ventas de helados (y)
data = {
    'Temperatura': [51, 52, 67, 65, 70, 69, 72, 75, 73, 81, 78, 83],
    'Ventas': [1, 0, 14, 14, 23, 20, 23, 26, 22, 30, 26, 36]
}

# Convertir a DataFrame
df = pd.DataFrame(data)


# Dividir los datos
X = df[['Temperatura']]  # Característica
y = df['Ventas']  # Etiqueta

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)


# Hacer predicciones
y_pred = model.predict(X_valid)


# Calcular métricas de evaluación
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')


# Visualizar resultados
plt.scatter(X, y, color='blue', label='Datos Reales')
plt.scatter(X_valid, y_pred, color='red', label='Predicciones')
plt.plot(X_train, model.predict(X_train), color='green', label='Línea de Regresión')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Ventas de Helados')
plt.title('Predicción de Ventas de Helados')
plt.legend()
plt.show()
