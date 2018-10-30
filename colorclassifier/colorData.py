import json 
import numpy as np
import keras
import tensorflowjs as tfjs
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

lableList = ['red-ish','green-ish','blue-ish','orange-ish','yellow-ish','pink-ish','purple-ish','brown-ish','grey-ish']

with open('./colorData.json') as f:
    data = json.load(f)
    entries = data['entries']
    size = len(entries)
    print(str(size))

xs = np.empty((0,3),int) #create two dimentional array using numpy
labels = np.array( [] )
for record in entries:
    xs = np.append(xs,np.array([[record['r']/255,record['g']/255,record['b']/255]]), axis=0)
    lbindex = lableList.index(record['label'])
    labels = np.append(labels,lbindex)

ys = np_utils.to_categorical(labels, num_classes=9)

#print(xs)
#print(ys)

model = Sequential()
model.add(Dense(units=16, activation='sigmoid', input_dim=3))
model.add(Dense(units=9, activation='softmax'))

sdg = keras.optimizers.SGD(lr=0.2)

model.compile(loss='categorical_crossentropy',optimizer=sdg)
model.fit(xs, ys, epochs=100, validation_split=0.1, shuffle=True)

model.save("Keras-16x1-100epoch")
tfjs.converters.save_keras_model(model, "tfjsv3")