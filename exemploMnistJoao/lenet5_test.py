import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('LeNet5.keras')

# Load and preprocess the test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = to_categorical(y_test, 10)

# Make predictions on the test data
predictions = model.predict(x_test)

# Collect errors
errors = []

for j in range(len(predictions)):
    if np.argmax(predictions[j]) != np.argmax(y_test[j]):
        errors.append((np.argmax(predictions[j]), np.argmax(y_test[j])))

# Initialize counts for each class and confusion class
class_errors = {i: {} for i in range(10)}

# Count errors by each class and their predicted classes
for pred, true in errors:
    if pred in class_errors[true]:
        class_errors[true][pred] += 1
    else:
        class_errors[true][pred] = 1


plt.figure(figsize=(15, 8))

for true_label in range(10):
    if true_label in class_errors:
        predicted_labels = list(class_errors[true_label].keys())
        error_counts = list(class_errors[true_label].values())
        plt.bar([str(label) for label in predicted_labels], error_counts, label=f'True Label {true_label}')

plt.xlabel('Classes preditas')
plt.ylabel('Número de erros')
plt.title('Número de erros X Classes preditas')
plt.legend()
plt.show()


