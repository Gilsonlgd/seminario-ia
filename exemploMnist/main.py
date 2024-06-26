import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Carregar o modelo treinado
model = load_model('mnist_cnn_model.keras')

# Função para carregar imagens de um diretório e fazer previsões
def load_and_predict(image_directory, model):
    images = []
    filenames = os.listdir(image_directory)
    
    for filename in filenames:
        img_path = os.path.join(image_directory, filename)
        img = image.load_img(img_path, color_mode='grayscale')
        img = img.resize((28, 28))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        images.append(img_array)
    
    images = np.vstack(images)
    images /= 255.0
    
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return filenames, predicted_classes

# Diretório onde estão as imagens de teste
image_directory = '../images/digits'

# Fazer previsões
filenames, predicted_classes = load_and_predict(image_directory, model)

# Exibir os resultados
for filename, predicted_class in zip(filenames, predicted_classes):
    print(f"Imagem {filename} foi classificada como dígito {predicted_class}.")
