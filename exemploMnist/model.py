import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Carregar dados de treino e teste do MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados de imagem para o intervalo [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Adicionar uma dimensão de canal para as imagens (formato para CNN)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Codificar os rótulos de classe para categorias (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definir o modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #Convolui a imagem com 32 filtros de tamanho 3x3
    MaxPooling2D((2, 2)),                                           #Reduz as dimensões da imagem pela metade, maxPooling em 2x2 
    Conv2D(64, (3, 3), activation='relu'),                          #Aplica 64 filtros de tamanho 3x3 na saída da camada de pooling anterior.
    MaxPooling2D((2, 2)),                                           #Reduz as dimensões da imagem pela metade, maxPooling em 2x2
    Flatten(),                                                      #Converte a saída 2D das camadas anteriores em um vetor 1D para ser usado pelas camadas
    Dense(128, activation='relu'),                                  #Camada totalmente conectada com 128 neurônios e ativação ReLU.
    Dropout(0.5),                                                   #Dropout de 50% para evitar overfitting
    Dense(10, activation='softmax')                                 #Camada de saída com 10 neurônios (um para cada classe) e ativação softmax.
])

# Compilar o modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
# 5, 7, 10 epochs
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Salvar o modelo treinado
model.save('mnist_cnn_model.keras')
print("Modelo salvo com sucesso.")

# Dificuldades no 0, 6 e 9
