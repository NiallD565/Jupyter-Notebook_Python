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

# For encoding categorical variables and pre processing.
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
from keras.layers import Dropout 

# Start a neural network, building it by layers.
model = kr.models.Sequential()
# model.add(kr.layers.Flatten())
# Add a hidden layer with 750 neurons.
model.add(kr.layers.Dense(units=784, activation='relu', input_dim=784))
# Add a hidden layer with 455 neurons.
model.add(kr.layers.Dense(units=455, activation='relu'))
# Add a hidden layer with 250 neurons.
model.add(kr.layers.Dense(units=250, activation='relu'))
# Add a hidden layer with 170 neurons.
model.add(kr.layers.Dense(units=170, activation='softplus'))
# Add a hidden layer with 120 neurons.
model.add(kr.layers.Dense(units=120, activation='linear'))
# Add a hidden layer with 50 neurons.
model.add(kr.layers.Dense(units=50, activation='relu'))
# Add a dropout layer every 1 in 5.
model.add(Dropout(0.2))

# Add a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Number of Epoch is the amount of times the training set is put through the model
# The batch size is the amount of images the models processes at one time
model.fit(inputs, outputs, epochs=10, batch_size=100)

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()
    
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

print((encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum())

def retrainNN():
	model.fit(inputs, outputs, epochs=10, batch_size=100)
	print((encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum())

def trainNNCustom(numEpc, btchSz):
	model.fit(inputs, outputs, epochs=numEpc, batch_size=btchSz)


def importImage():
	imageName = input("Enter the file name: ")
	userInput = input("Enter the label for the image uploaded: ")
	print("The label for this image is: ", userInput)
	# Used to open image file using pillow
	from PIL import Image
	temp = Image.open("Images/" + imageName + ".png").convert('L')
	# Convert to a 1 dimensioanl array with 784 nodes
	temp = temp.resize((28, 28))
	imgArray = np.array(temp)
	# converts the array to a a 1D array of 784 nodes
	imgArray = imgArray.reshape(1, 784)

	prediction = model.predict(imgArray)
	#print(prediction)
	print("Prediction: ", prediction.argmax(axis=1))	

option =int(input("1 to retrain model: \n2 to read an image: \n3 to train the neural netork with custom epochs and batch size \n0 to Quit:"))
while option != 0:
	if option==1:
		retrainNN()
		option = input("1 to retrain model: \n2 to read an image: \n3 to train the neural netork with custom epochs and batch size \n 0 to Quit:")
	elif option==2:
		importImage()
		option = input("1 to retrain model: \n2 to read an image: \n3 to train the neural netork with custom epochs and batch size \n 0 to Quit:")
	elif option==3:
		numEpc = input("Enter the number of epochs: ")
		btchSz = input("Enter batch size: ")
		trainNNCustom(int(numEpc), int(btchSz))
		option = input("1 to retrain model: \n2 to read an image: \n3 to train the neural netork with custom epochs and batch size \n 0 to Quit:")
	else:
		exit()	
