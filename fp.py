from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Cargar el conjunto de datos de viviendas de California
housing = fetch_california_housing()
X = housing.data      # Características de entrada
y = housing.target    # Precio de la vivienda (etiqueta)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos (normalización) para mejorar el rendimiento de la red neuronal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dividir el conjunto de entrenamiento en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Función para graficar el historial de entrenamiento
def plot_history(history, title):
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title(title)
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.show()

# Configuración 1: Red profunda con más capas, menos neuronas
def build_model1():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Configuración 2: Red con menos capas, más neuronas por capa
def build_model2():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Configuración 3: Red más simple con `tanh`
def build_model3():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Configuración 4: Red con sólo dos capas y menos neuronas
def build_model4():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Entrenar y evaluar cada modelo
models = [build_model1(), build_model2(), build_model3(), build_model4()]
histories = []
mse_scores = []

for i, model in enumerate(models):
    print(f"\nEntrenando el Modelo {i + 1}")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=10, verbose=0)
    histories.append(history)
    
    # Evaluar el modelo
    mse = model.evaluate(X_test, y_test, verbose=0)
    mse_scores.append(mse)
    print(f'Modelo {i + 1} - MSE en el conjunto de prueba: {mse}')

    # Graficar la pérdida
    plot_history(history, f"Modelo {i + 1}: Pérdida durante el entrenamiento")

# Comparar los resultados
for i, mse in enumerate(mse_scores):
    print(f"Modelo {i + 1} - MSE: {mse}")
