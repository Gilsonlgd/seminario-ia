from tensorflow.keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical

# Carregar os dados
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255

# Carregar o modelo treinado
model = load_model('cifar10_cnn_model.keras')

# Transformar os rótulos de teste em one-hot encoding
num_classes = 10  # Número de classes no CIFAR-10
y_test = to_categorical(y_test, num_classes)

# Avaliar o modelo
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])