import pandas as pd
import os
import cv2
import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers.core import Dropout

path = 'D:\MaDoc\GTSRB\Final_Training\Images'
pixels = []
labels = []
for dir in os.listdir(path):
    print(dir)
    class_file = os.path.join(path, dir)
    print(class_file)
    infor_class = os.path.join(class_file, 'GT-' + dir + '.csv')
    print(infor_class)
    dataset = pd.read_csv(infor_class, sep=';')
    for row in dataset.iterrows():
        pixel = cv2.imread(os.path.join(class_file, row[1].Filename))
        pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'],
                      row[1]['Roi.Y1']:row[1]['Roi.Y2']]
        pixel = cv2.resize(pixel, (64, 64))

        pixels.append(pixel)
        labels.append(row[1]['ClassId'])

pixels = np.array(pixels)
labels = np.array(labels)
labels = to_categorical(labels)
print(pixels.shape)
print(labels.shape)
randomize = np.arange(len(pixels))
np.random.shuffle(randomize)
X = pixels[randomize]
Y = labels[randomize]

train_size = int(X.shape[0] * 0.6)
X_train, X_val_test = X[:train_size], X[train_size:]
y_train, y_val_test = Y[:train_size], Y[train_size:]

test_size = int(X_val_test.shape[0]*0.5)
X_val, X_test = X_val_test[:test_size], X_val_test[test_size:]
y_val, y_test = y_val_test[:test_size], y_val_test[test_size:]
filter_size = (3, 3)
pool_size = (2, 2)

model = Sequential()
model.add(Conv2D(16, filter_size, activation='relu',
          input_shape=(64, 64, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(16, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Conv2D(32, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Conv2D(64, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=16,
          validation_data=(X_val, y_val))

X_new = X_test[78]
y_new = y_test[78]

y_pre = model.predict(X_test[78:79])
y_pre = np.argmax(y_pre, axis=1)
plt.imshow(X_new)
print(y_pre)
print(y_new)
plt.show()
