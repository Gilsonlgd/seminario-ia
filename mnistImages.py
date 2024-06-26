import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# Carregar dados de treino e teste do MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Mostra as primeiras 3 imagens do conjunto de treino
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(15, 15))
for i in range(3):
    axs[i].imshow(x_train[i], cmap='gray')
    axs[i].axis('off')

plt.show()

# Contar as amostras de cada classe no conjunto de treino
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))

# Mostrar a distribuição das classes
for label, count in class_distribution.items():
    print(f'Classe {label}: {count} amostras')

