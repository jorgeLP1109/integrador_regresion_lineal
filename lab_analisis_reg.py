import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import io

# Datos CSV proporcionados
data = """30,Hombre,1,0,1,0
25,Mujer,0,1,0,1
40,Hombre,0,0,1,0
35,Mujer,1,1,0,1
45,Hombre,1,1,1,0
50,Mujer,0,0,0,0
"""

# Crear DataFrame a partir de los datos CSV
df = pd.read_csv(io.StringIO(data), header=None, names=['edad', 'sexo', 'anemico', 'diabetico', 'fumador', 'muerto'])

# Convertir la columna 'sexo' a variables dummy
df = pd.get_dummies(df, columns=['sexo'], drop_first=True)

# Eliminar las filas con valores faltantes en la columna 'edad'
df = df.dropna(subset=['edad'])

# Separar las edades conocidas y desconocidas
edades_conocidas = df.loc[:, 'edad']
X_conocidas = df.drop(['edad'], axis=1)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_conocidas, edades_conocidas, test_size=0.2, random_state=42)

# Inicializar y ajustar el modelo de regresión lineal
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

# Predecir las edades en el conjunto de prueba
edades_predichas = modelo_regresion.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, edades_predichas)

print("\nResultados:")
print(f"Edades reales en el conjunto de prueba: {y_test.values}")
print(f"Edades predichas en el conjunto de prueba: {edades_predichas}")
print(f"Error Cuadrático Medio: {mse}")
