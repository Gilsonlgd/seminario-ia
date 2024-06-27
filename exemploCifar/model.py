#https://medium.com/neuronio-br/entendendo-redes-convolucionais-cnns-d10359f21184

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10


# Carrega os dados do CIFAR-10
batch_size = 32 
n_classes = 10 
epochs = 40
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Guarda a altura e largura das imagens
height = x_train.shape[1]
width = x_train.shape[2]

# Separa 5000 imagens para validação
x_val = x_train[:5000,:,:,:]
y_val = y_train[:5000]

# Pega o resto para treino
x_train = x_train[5000:,:,:,:]
y_train = y_train[5000:]

print('Training dataset: ', x_train.shape, y_train.shape)
print('Validation dataset: ', x_val.shape, y_val.shape)
print('Test dataset: ', x_test.shape, y_test.shape)

# Convertendo os rótulos para one-hot encoding
y_train = to_categorical(y_train, n_classes)
y_val = to_categorical(y_val, n_classes)
y_test = to_categorical(y_test, n_classes)

# Normalizando as imagens
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255

# Definindo o modelo
def create_model():
  model = Sequential()
  model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(height, width, 3), strides=1, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(1,1)))
  model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(1,1)))
  model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(1,1)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(n_classes, activation='softmax'))
  return model

def optimizer():
    return SGD(learning_rate=1e-2)

model = create_model()
model.compile(optimizer=optimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val,y_val),verbose=1)

model.save('cifar10_cnn_model.keras')