import util
import numpy as np
import zipfile
import os
import math

## Constants
DATUM_WIDTH_FACE = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

class Neural_Network:
 
  def __init__(self, data,width,height):
    """
    Create a new datum from file input (standard MNIST encoding).
    """
    DATUM_HEIGHT = height
    DATUM_WIDTH=width
    self.height = DATUM_HEIGHT
    self.width = DATUM_WIDTH
    self.weight_faces = None
    if data == None:
      data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)] 
    self.pixels = data
    
  def getPixel(self, column, row):
    """
    Returns the value of the pixel at column, row as 0, or 1.
    """
    return self.pixels[column][row]
      
  def getPixels(self):
    """
    Returns all pixels as a list of lists.
    """
    return self.pixels    
      
# Data processing, cleanup and display functions  
def loadDataFile(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.
    
    (Return less than n items if the end of file is encountered).
    """
    DATUM_WIDTH = width
    DATUM_HEIGHT = height
    fin = readlines(filename)
    fin.reverse()
    items = []
    count = 0
    for i in range(n):
        data = []
        for j in range(height):
            # Read a line from the file
            line = fin.pop()
            # Convert symbols to 0s and 1s
            #print(line)
            #print(list(map(convertToInteger, line)))
            data.append(list(map(convertToInteger, line)))
        if len(data[0]) < DATUM_WIDTH - 1:
            # We encountered the end of the file
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Neural_Network(data, DATUM_WIDTH, DATUM_HEIGHT))
        count = + 1
    print(count)
    return items


def readlines(filename):
  "Opens a file or reads it from the zip archive data.zip"
  if(os.path.exists(filename)): 
    return [l[:-1] for l in open(filename).readlines()]
  else: 
    z = zipfile.ZipFile('data.zip')
    return z.read(filename).split('\n')
    
def loadLabelsFile(filename, n):
  """
  Reads n labels from a file and returns a list of integers.
  """
  fin = readlines(filename)
  labels = []
  for line in fin[:min(n, len(fin))]:
    if line == '':
        break
    labels.append(int(line))
  return labels  
    
def IntegerConversionFunction(character):
  """
  Helper function for file reading.
  """
  if(character == ' '):
    return 0
  elif(character == '+'):
    return 1
  elif(character == '#'):
    return 2    

def convertToInteger(data):
  """
  Helper function for file reading.
  """
  if type(data) != type([]):
    return IntegerConversionFunction(data)
  else:
    return map(convertToInteger, data)

def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(output):
    return sigmoid_activation(output) * (1 - sigmoid_activation(output))

def backpropagate(sample, predicted, actual, hidden_layer_values, output_weights, hidden_weights):
    # Calculate the error at the output layer

    hidden_layer_values = np.array(hidden_layer_values)
    learning_rate = 0.01
    error = predicted - actual
    sample = sample.reshape(1, -1)

    # calculate the output gradient
    output_gradient = hidden_layer_values * error

    # calculate the hidden layer gradient
    #print(output_weights.shape)
    #print(hidden_layer_values.shape)
    hidden_out = ((output_weights.T * error) * sigmoid_derivative(hidden_layer_values))
    hidden_out = hidden_out.reshape((1000, 1))
    hidden_gradient = hidden_out * sample
    
    # update the weights
    output_weights -= learning_rate * output_gradient
    hidden_weights -= learning_rate * hidden_gradient

    #print(output_weights.shape)
    

    return output_weights, hidden_weights

def nn_face(n):
  
  epochs = 5
  data = loadDataFile('data/facedata/facedatatrain', 451, 60, 70)
  labels = loadLabelsFile('data/facedata/facedatatrainlabels', 451)
  hidden_weights = np.random.uniform(low=-10, high=10, size=(1000, 4200))  # we need different weights for each of the nodes on the hidden layer
  output_weights = np.random.uniform(low=-10, high=10, size=1000)
  bias = 1


  num_samples = int(n * 451)


  for epoch in range(epochs):
    for _ in range(num_samples):
      
      #print(f'Before Reset: {len(hidden_layer_values)}')
      hidden_layer_values = []

      idx = np.random.randint(0, 451)
      sample = data[idx]
      sample = np.array(sample.getPixels()).flatten()

      total_sum = 0
      for node in range(1000):
          total_sum = np.dot(hidden_weights[node], sample) 
          hidden_layer_values.append(sigmoid_activation(total_sum + bias))
      
      final_output = np.sum(output_weights * hidden_layer_values)
      
      predicted = sigmoid_activation(final_output + bias)
      actual = labels[idx]

      #print(len(hidden_layer_values))
      #print(output_weights.shape)

      if predicted < 0.5 and actual == 1 :
          #print("About to backpropagate.")
          output_weights, hidden_weights = backpropagate(sample, predicted, actual, hidden_layer_values, output_weights, hidden_weights)
          #print("Finished backpropagating.")
      elif predicted > 0.5 and actual == 0:
          #print("About to backpropagate.")
          output_weights, hidden_weights= backpropagate(sample, predicted, actual, hidden_layer_values, output_weights, hidden_weights)
          #print("Finished backpropagating.")          

  # now test the test data to see how accurate it is    
  data_test = loadDataFile('data/facedata/facedatavalidation', 301, 60, 70)
  labels_test = loadLabelsFile('data/facedata/facedatavalidationlabels', 301)
  
  accuracies = []

  for idx in range(301):
      hidden_layer_values = []

      sample = data_test[idx]
      sample = np.array(sample.getPixels()).flatten()

      total_sum = 0
      for node in range(1000):
          total_sum = np.dot(hidden_weights[node], sample) 
          hidden_layer_values.append(sigmoid_activation(total_sum + bias))
      
      final_output = np.sum(output_weights * hidden_layer_values)
      
      predicted = sigmoid_activation(final_output + bias)
      actual = labels_test[idx]

      if (predicted > 0.5 and actual == 1) or (predicted < 0.5 and actual == 0): 
         accuracies.append(1)
      else:
         accuracies.append(0)

  accuracy_mean = np.mean(accuracies)
  accuracy_std = np.std(accuracies)

  return accuracy_mean, accuracy_std

def nn_digit(n):
  
  epochs = 10
  data = loadDataFile('data/digitdata/trainingimages', 451, 28, 28)
  labels = loadLabelsFile('data/digitdata/traininglabels', 451)
  hidden_weights = np.random.randint(low= -400, high= 400, size=(256, 28, 28)) # we need different weights for each of the nodes on the hidden layer
  output_weights = np.random.randint(low= -400, high= 400, size=(10, 256))
  hidden_biases = np.random.randint(low=-200, high=200, size=256)
  output_biases = np.random.randint(low = -200, high = 200, size = 10)
  hidden_layer_values = [] 
  num_samples = int(n * 451)

    # when creating the neural network we need the following:
        # the input layer (this layer will be the individual pixels of the image)
        # the hidden layer (this layer will use the sigmoid function g(Z) = (1/(1 + e^-2)))
        # the output layer (this layer is computed using sigmoid of each of the values computed in the hidden layer)

  for epoch in range(epochs):

    for _ in num_samples:
      
      idx = np.random.randint(0, 451)
      sample = data[idx]
      
      for node in range(256):
        total_sum = 0
        for i in range(28):
          for j in range(28):
              total_sum += hidden_weights[node][i][j] * sample.getPixel(i, j)

        hidden_layer_values.append(sigmoid_activation(total_sum + hidden_biases[node]))    
        

      final_output = np.dot(output_weights, hidden_layer_values) + output_biases
      digit_predicted = np.argmax(final_output)
      
      
def _test():
  average, std = nn_face(.1)
  print(average, std)

if __name__ == "__main__":
  _test()  