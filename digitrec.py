# For unzipping the file within the script.
import gzip
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content_images = f.read()
    
# For unzipping the file within the script.
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    file_content_labels = f.read()
	
import numpy as np
image = ~np.array(list(file_content_images[16:800])).reshape(28,28).astype(np.uint8)

with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()
    
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)/255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

# For encoding categorical variables.
import sklearn.preprocessing as pre

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

outputs[0]

inputs = train_img.reshape(60000, 784)

# ------- MODEL -------
# Import keras.
import keras as kr
# Import Tensorflow
import tensorflow as tf
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
# Start a neural network, building it by layers.
model = kr.models.Sequential()
# model.add(kr.layers.Flatten())
# Add a hidden layer with 300 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=650, activation='relu'))
# Add a hidden layer with 325 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=325, activation='sigmoid'))
# Add a hidden layer with 150 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=150, activation='relu'))
# Add a hidden layer with 50 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=50, activation='relu'))
# Add a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Don't run this unless you really want to
model.fit(inputs, outputs, epochs=5, batch_size=15)