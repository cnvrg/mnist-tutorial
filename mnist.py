'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''


from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import argparse
import os

if not os.path.exists(os.path.dirname("output/")):
    os.makedirs(os.path.dirname("output/"))

parser = argparse.ArgumentParser(description='Running MNIST Algorithm')
parser.add_argument('--epochs', help='number of epochs to run', default='12')
parser.add_argument('--batch_size', help='iteration batch size', default='128')
args = parser.parse_args()

num_classes = 10
batch_size = int(args.batch_size)
epochs = int(args.epochs)

print('cnvrg_tag_batch_size:', batch_size)
print('cnvrg_tag_num_classes:', num_classes)
print('cnvrg_tag_epochs:', epochs)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('cnvrg_tag_x_train_shape:', x_train.shape)
print('cnvrg_tag_train_samples:', x_train.shape[0])
print('cnvrg_tag_test_samples:', x_test.shape[0])

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


tbCallback = keras.callbacks.TensorBoard(histogram_freq=0, write_graph=True, write_images=True)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[tbCallback])

score = model.evaluate(x_test, y_test, verbose=0)
print('cnvrg_tag_test_loss:', score[0])
print('cnvrg_tag_test_accuracy:', score[1])
model.save('output/mnist_model.h5')