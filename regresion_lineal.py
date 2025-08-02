from sklearn.linear_model import LinearRegression
import numpy as np

# Datos: experiencia vs salario
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([30000, 35000, 40000, 45000, 50000])

modelo = LinearRegression()
modelo.fit(x, y)

# Predecir salario con 6 años de experiencia
salario = modelo.predict([[6]])
print(f"Salario estimado con 6 años de experiencia: ${salario[0]:.2f}")
