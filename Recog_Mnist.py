from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from numpy.lib import loadtxt
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import random
from keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model = Sequential()
# model.add(Conv2D(64, input_shape=(28, 28, 1), kernel_size=(
#     3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(32, activation='relu', kernel_size=(3, 3), padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))
# model.summary()

# model.compile(loss='categorical_crossentropy',
#               metrics=['accuracy'], optimizer='adam')
# model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

# model.save('Mnist.h5')
model = load_model('Mnist.h5')

X_new = X_test[164]
y_new = y_test[164]

y_pre = model.predict(X_test[164:165])
y_pre = np.argmax(y_pre, axis=1)
print(y_pre)
print(y_new)
plt.imshow(X_new)
plt.show()
