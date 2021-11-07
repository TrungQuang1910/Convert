from numpy import loadtxt
import numpy
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model

path = 'D:\\MaDoc\\tieuduong.csv'
dataset = loadtxt(path, delimiter=',')

X = dataset[:, 0:8]
Y = dataset[:, 8]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, Y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2)

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=[
              'accuracy'], optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=8,
          validation_data=(X_val, y_val))

model.save('TrungDataset.h5')
model = load_model('TrungDataset.h5')

X_new = X_test[34]
y_rel = y_test[34]
X_new = numpy.expand_dims(X_new, axis=0)
print(X_new)
y_pre = model.predict(X_new)
print(y_pre)
print(y_rel)
