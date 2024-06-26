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

def load(f, label):
    # load the file into tensor
    image = tf.io.read_file(f)
    # Decode it to JPEG format
    image = tf.image.decode_jpeg(image)
    # Convert it to tf.float32
    image = tf.cast(image, tf.float32)
    
    return image, label

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

def load_image_train(image_file, label):
    image, label = load(image_file, label)
    image = random_jitter(image)
    image = normalize(image)
    return image, label

def load_image_val(image_file, label):
    image, label = load(image_file, label)
    image = central_crop(image)
    image = normalize(image)
    return image, label

# Carrega os dados de treinamento
temp_ds = tf.data.Dataset.list_files(os.path.join('./dataset/my_cat_dog', 'train', 'cat', '*.jpg'))
temp_ds = temp_ds.map(lambda x: (x, 0))

temp2_ds = tf.data.Dataset.list_files(os.path.join('./dataset/my_cat_dog', 'train', 'dog', '*.jpg'))
temp2_ds = temp2_ds.map(lambda x: (x, 1))

train_ds = temp_ds.concatenate(temp2_ds)

# Embaralha os dados de treinamento
buffer_size = tf.data.experimental.cardinality(train_ds).numpy()
train_ds = train_ds.shuffle(buffer_size)\
                   .map(load_image_train, num_parallel_calls=16)\
                   .batch(20)\
                   .repeat()

# Carrega os dados de validação
temp_ds = tf.data.Dataset.list_files(os.path.join('./dataset/my_cat_dog/', 'val', 'cat', '*.jpg'))
temp_ds = temp_ds.map(lambda x: (x, 0))

temp2_ds = tf.data.Dataset.list_files(os.path.join('./dataset/my_cat_dog', 'val', 'dog', '*.jpg'))
temp2_ds = temp2_ds.map(lambda x: (x, 1))

val_ds = temp_ds.concatenate(temp2_ds)

val_ds = val_ds.map(load_image_val, num_parallel_calls=16)\
               .batch(20)\
               .repeat()


# Carrega os dados de teste
temp_ds = tf.data.Dataset.list_files(os.path.join('./dataset/my_cat_dog', 'test', 'cat', '*.jpg'))
temp_ds = temp_ds.map(lambda x: (x, 0))

temp2_ds = tf.data.Dataset.list_files(os.path.join('./dataset/my_cat_dog', 'test', 'dog', '*.jpg'))
temp2_ds = temp2_ds.map(lambda x: (x, 1))

test_ds = temp_ds.concatenate(temp2_ds)

test_ds = test_ds.map(load_image_val, num_parallel_calls=16)\
                .shuffle(buffer_size)\
               .batch(20)\
               .repeat()

train_len = len(glob(os.path.join('./dataset/my_cat_dog', 'train', 'cat', '*.jpg'))) * 2
val_len = len(glob(os.path.join('./datasets/my_cat_dog', 'val', 'cat', '*.jpg'))) * 2
test_len = len(glob(os.path.join('./dataset/my_cat_dog', 'test', 'cat', '*.jpg'))) * 2

print(train_len, val_len, test_len)

for images, labels in train_ds.take(1):
    fig, ax = plt.subplots(1, 10, figsize=(20, 6))
    for j in range(10):
        image = images[j].numpy()
        image = image / np.amax(image)
        image = np.clip(image, 0, 1)
        ax[j].imshow(image)
        ax[j].set_title(labels[j].numpy())
plt.show()


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
model.summary()

train_len = len(glob(os.path.join('../datasets/my_cat_dog', 'train', 'cat', '*.jpg'))) * 2
val_len = len(glob(os.path.join('../datasets/my_cat_dog', 'val', 'cat', '*.jpg'))) * 2
test_len = len(glob(os.path.join('../datasets/my_cat_dog', 'test', 'cat', '*.jpg'))) * 2

print(train_len, val_len, test_len)

model.fit(train_ds, steps_per_epoch=train_len // 20,
          validation_data=val_ds,
          validation_steps=val_len//20,
          epochs=30,
          callbacks=[cp_callback]
          )

model.evaluate(test_ds, steps=test_len//20)

model.save('cat_dog_cnn.keras')