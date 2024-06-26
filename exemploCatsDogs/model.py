import os
import time
import re
from glob import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image

print("Tensorflow: v{}".format(tf.__version__))
# %matplotlib inline
IMAGE_SIZE = (150, 150)

def load(file_path, label):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # Decode the image as JPEG format
    img = tf.image.decode_image(img, channels=3)
    # Resize the image to the desired size
    img.set_shape([None, None, 3]) 
    img = tf.image.resize(img, IMAGE_SIZE)
    # Convert the image to float32 type
    img = tf.cast(img, tf.float32)
    # Normalize the pixel values to the range [0, 1]
    img = img / 255.0
    return img, label

def filter_invalid_images(file_path, label):
    # Check if the image can be decoded correctly
    try:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return True
    except:
        return False

def filter_invalid_images(image, label):
    return tf.not_equal(label, -1)

def resize(input_image, size):
    return tf.image.resize(input_image, size)

def random_crop(input_image):
    return tf.image.random_crop(input_image, size=[150, 150, 3])

def central_crop(input_image):
    image = resize(input_image, [176, 176])
    return tf.image.central_crop(image, central_fraction=0.84)

def random_rotation(input_image):
    angles = np.random.randint(0, 3, 1)
    return tf.image.rot90(input_image, k=angles[0])

def random_jitter(input_image):
    # Resize it to 176 x 176 x 3
    image = resize(input_image, [176, 176])
    # Randomly Crop to 150 x 150 x 3
    image = random_crop(image)
    # Randomly rotation
    image = random_rotation(image)
    # Randomly mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def normalize(input_image):
    mid = (tf.reduce_max(input_image) + tf.reduce_min(input_image)) / 2
    input_image = input_image / mid - 1
    return input_image


temp_ds = tf.data.Dataset.list_files(os.path.join('../datasets/pet-images/Cat', '*.jpg'))
temp_ds = temp_ds.map(lambda x: (x, 0))

temp2_ds = tf.data.Dataset.list_files(os.path.join('../datasets/pet-images/Dog', '*.jpg'))
temp2_ds = temp2_ds.map(lambda x: (x, 1))

# separa em treinamento, validação e teste
train_size = int(0.7 * 2000)
val_size = int(0.15 * 2000)
test_size = int(0.15 * 2000)

train_ds = temp_ds.take(train_size)
val_ds = temp_ds.skip(train_size).take(val_size)
test_ds = temp_ds.skip(train_size + val_size).take(test_size)

train_ds = train_ds.concatenate(temp2_ds.take(train_size))
val_ds = val_ds.concatenate(temp2_ds.skip(train_size).take(val_size))
test_ds = test_ds.concatenate(temp2_ds.skip(train_size + val_size).take(test_size))

# Embaralha os dados de treinamento
buffer_size = tf.data.experimental.cardinality(train_ds).numpy()
train_ds = train_ds.shuffle(buffer_size)\
                   .map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                   .filter(filter_invalid_images)\
                   .batch(20)\
                   .repeat()

val_ds = temp_ds.concatenate(temp2_ds)
val_ds = val_ds.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
               .filter(filter_invalid_images)\
               .batch(20)\
               .repeat()

test_ds = temp_ds.concatenate(temp2_ds)
test_ds = test_ds.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                 .filter(filter_invalid_images)\
                 .batch(20)\
                 .repeat()


class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(Conv, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
model = tf.keras.Sequential(name='Cat_Dog_CNN')

model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 3)))
model.add(Conv(filters=32, kernel_size=(3, 3)))
model.add(Conv(filters=64, kernel_size=(3, 3)))
model.add(Conv(filters=128, kernel_size=(3, 3)))
model.add(Conv(filters=128, kernel_size=(3, 3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax))


checkpoint_path = "./train/cat_dog_cnn/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


train_len = train_size * 2
val_len = val_size * 2
test_len = test_size * 2

model.fit(train_ds, 
          steps_per_epoch=int(train_len / 20),
          validation_data=val_ds,
          validation_steps=int(val_len / 20),
          epochs=30,
          callbacks=[cp_callback])

model.evaluate(test_ds, steps=int(test_len / 20))
model.save('cat_dog_cnn_model.keras')