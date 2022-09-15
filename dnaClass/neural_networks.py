import numpy as np

np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

model = Sequential()
model.add(Dense(5, input_dim=inp_n, activation='sigmoid'))
model.add(Dropout(0.005, noise_shape=None, seed=None))
#model.add(Dense(80, activation='relu'))
#model.add(Dropout(0.3, noise_shape=None, seed=None))

#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train[train], y_train[train], epochs=220, batch_size=12)
predictions = model.predict(X_train[test])
scores = model.evaluate(X_train[test], y_train[test])
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



model = Sequential()
model.add(Dense(5, input_dim=inp_n, activation='sigmoid'))
model.add(Dropout(0.005, noise_shape=None, seed=None))

#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt = optimizers.SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.33,epochs=220, batch_size=12)
# validation_split=0.00,
#predictions = model.predict(X_test)
#scores = model.evaluate(X_test, y_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'],color='black',linewidth = 3)
plt.plot(history.history['val_acc'],color='green',linewidth = 3)
plt.title('NN Model Accuracy',size='x-large')
plt.ylabel('Accuracy',size='large')
plt.xlabel('epoch',size='large')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'],color='black',linewidth = 3)
plt.plot(history.history['val_loss'],color='green',linewidth = 3)
plt.title('NN Model Loss',size='x-large')
plt.ylabel('Loss',size='large')
plt.xlabel('epoch',size='large')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
