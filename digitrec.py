# Adapted from https://docs.python.org/3/library/gzip.html

# For unzipping the file within the script.
import gzip
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content_images = f.read()
    
# Adapted from: https://docs.python.org/2/library/gzip.html
# For unzipping the file within the script.
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    file_content_labels = f.read()
	
	
# %matplotlib inline
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
# Import keras.
import keras as kr
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation

image = ~np.array(list(file_content_images[16:800])).reshape(28,28).astype(np.uint8)

# Start a neural network, building it by layers.
model = kr.models.Sequential()

# Add a hidden layer with 2000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784))
# Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=750, activation='relu', input_dim=784))
# Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=250, activation='relu', input_dim=784))
# Dropout drops random biases within the network as it trains
model.add(Dropout(0.2))
# Add a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()
    
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

inputs = train_img.reshape(60000, 784)

# For encoding categorical variables.
import sklearn.preprocessing as pre

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

# Don't run this unless you really want to
model.fit(inputs, outputs, epochs=10, batch_size=15)