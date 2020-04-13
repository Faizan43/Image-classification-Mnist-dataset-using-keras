from keras.preprocessing.image import load_img,array_to_img
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

(X_train,y_train),(X_test,y_test)=mnist.load_data()
width,height=28,28
X_train=X_train.reshape(60000,width*height)
X_test=X_test.reshape(10000,width*height)


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255.0
X_test/=255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

model=Sequential()
model.add(Dense(512, activation='relu' , input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'],)
model.summary()

trained_model=model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))
plt.plot(trained_model.history['acc'])
plt.plot(trained_model.trained_model['val_acc'])

score=model.evaluate(X_test,y_test)

