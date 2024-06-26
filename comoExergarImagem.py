#import all required lib
import matplotlib.pyplot as plt

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

# Carrega a imagem e transforma para gray scale
image = imread('./images/digits/sample_1_digit.png')
image = rgb2gray(image)

# Redimensiona a imagem para 28x28 pixels
image = resize(image, (28,28), mode='reflect')

def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111)
visualize_input(image, ax)
plt.show()


