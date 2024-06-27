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

# Mostrando exemplos de imagens do dataset
cols=2
fig = plt.figure()
print('training:')
for i in range(5):
    a = fig.add_subplot(cols, int(np.ceil(n_classes/float(cols))), i + 1)
    img_num = np.random.randint(x_train.shape[0])
    image = x_train[i]
    id = y_train[i]
    plt.imshow(image)
    a.set_title(label_names[id[0]])
fig.set_size_inches(8,8)
plt.show()