# Setup
import pandas as pd
from keras.datasets import mnist
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# Load the MNIST dataset and preprocess the data (normalization, reshaping)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Checking Shape
x_train.data.shape 
print("X_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", x_test.shape)
print("y_test shape", y_test.shape)

# Viasualizing part of the data
plt.rcParams['figure.figsize'] = (9,9) # Make the figures a bit bigger

for i in range(9):
    plt.subplot(3,3,i+1)
    num = random.randint(0, len(x_train))
    plt.imshow(x_train[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[num]))

plt.tight_layout()

# Reshaping and normalize. Each pixel is an 8-bit integer from 0-255. 0 is full black, while 255 is full white.
# Instead of a 28 x 28 matrix, we build our network to accept a 784-length vector.
# We'll also normalize the inputs to be in the range [0-1]

x_train = x_train.reshape(60000, 784) # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
x_test = x_test.reshape(10000, 784)   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.

x_train = x_train.astype('float32')   # change integers to 32-bit floating point numbers
x_test = x_test.astype('float32')

x_train /= 255                        # normalize each value for each pixel for the entire vector for each input
x_test /= 255

print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)

# Building a basic ANN with at least one hidden layer.
classifier = Sequential()
classifier.add(Dense(512, input_shape=(784,), kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(512, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, epochs = 10)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

scores = classifier.evaluate(x_test, y_test)

print("Accuracy: %d%%" % (scores[1]*100))

x_train.shape[1]

# Train the ANN using an appropriate optimizer and activation functions, and evaluate its performance.
classifier_2 = Sequential()
classifier_2.add(Dense(32, input_shape=(784,), activation = 'relu'))
classifier_2.add(Dense(32, input_shape=(32,), activation = 'relu'))
classifier_2.add(Dense(units = 10, input_shape=(32,), activation = 'softmax'))

classifier_2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
classifier_2.fit(x_train, y_train, batch_size = 128, epochs = 75)

scores_2 = classifier_2.evaluate(x_test, y_test)
print("Accuracy: %d%%" % (scores_2[1]*100))
