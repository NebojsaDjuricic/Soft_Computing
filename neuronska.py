import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

(x_train, y_train),(x_test, y_test) = mnist.load_data()
                                                    # kanal boje
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# normalizacija
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# kreiranje modela
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation=tf.nn.relu),
    BatchNormalization(axis=-1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation=tf.nn.relu),
    BatchNormalization(axis=-1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation=tf.nn.relu),
    BatchNormalization(axis=-1),
    Dropout(0.5),
    Dense(10, activation=tf.nn.softmax)
])

model.summary()

model.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=6, shuffle=True, verbose=1)

rez = model.evaluate(x_test, y_test, verbose=1)
print('Preciznost: ', rez[1] * 100, '%')

model.save('neuronska.h5')