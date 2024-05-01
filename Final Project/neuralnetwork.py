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
    z = np.array(z)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(output):
    output = np.array(output)
    sigmoid = sigmoid_activation(output)
    return sigmoid * (1 - sigmoid)

def mse_loss_derivative(predicted, actual):
    return predicted - actual

def backprop(hidden_layer_weights, output_weights, hidden_biases, output_biases, hidden_layer_outputs, input_layer, actual, predicted):
    learning_rate = 0.01
    # 1. Calculate the output layer error derivative
    output_error = mse_loss_derivative(predicted, actual)
    # 2. Calculate the derivative of the loss w.r.t the output weights (gradient)
    delta_output = output_error * sigmoid_derivative(predicted)
    output_weights_gradient = np.outer(delta_output, hidden_layer_outputs)
    # 3. Calculate the derivative of the loss w.r.t the output bias
    output_bias_gradient = delta_output
    # 4. Propagate the error back to the hidden layer
    hidden_error = np.dot(output_weights.T, delta_output.reshape(-1, 1))
    hidden_delta = hidden_error * np.array([sigmoid_derivative(output) for output in hidden_layer_outputs])
    # 5. Calculate gradients for hidden layer weights
    hidden_weights_gradient = np.array([np.outer(delta, input_layer) for delta in hidden_delta])
    # 6. Calculate gradients for hidden layer biases
    hidden_biases_gradient = hidden_delta
    # 7. Update weights and biases
    output_weights -= learning_rate * output_weights_gradient
    hidden_layer_weights -= learning_rate * hidden_weights_gradient
    output_biases -= learning_rate * output_bias_gradient
    hidden_biases -= learning_rate * hidden_biases_gradient

    return hidden_layer_weights, output_weights, hidden_biases, output_biases
      
def nn_face(n):
  
  epochs = 10
  data = loadDataFile('data/facedata/facedatatrain', 451, 60, 70)
  labels = loadLabelsFile('data/facedata/facedatatrainlabels', 451)
  hidden_weights = np.random.uniform(low= -10, high= 10, size=(1000, 70, 60)) # we need different weights for each of the nodes on the hidden layer
  output_weights = np.random.uniform(low= -10, high= 10, size= 1000)
  hidden_biases = np.random.uniform(low=-10, high=10, size=1000)
  output_bias = np.random.uniform(low = -10, high = 10, size = 1)
  hidden_layer_values = [] 
  num_samples = int(n * 451)

    # when creating the neural network we need the following:
        # the input layer (this layer will be the individual pixels of the image)
        # the hidden layer (this layer will use the sigmoid function g(Z) = (1/(1 + e^-2)))
        # the output layer (this layer is computed using sigmoid of each of the values computed in the hidden layer)

  for epoch in range(epochs):

    for _ in range(num_samples):
      
      idx = np.random.randint(0, 451)
      sample = data[idx]
      
      for node in range(1000):
        total_sum = 0
        for i in range(70):
          for j in range(60):
              total_sum += hidden_weights[node][i][j] * sample.getPixel(i, j)

        hidden_layer_values.append(sigmoid_activation(total_sum + hidden_biases[node]))    
        
      final_output = 0
    
      for i in range(1000):
          final_output += output_weights[i] * hidden_layer_values[i]

      predicted = sigmoid_activation(final_output + output_bias)

      actual = labels[idx]

      if predicted > 0.5 and actual == 0:
          backprop(hidden_weights, output_weights, hidden_biases, output_bias, hidden_layer_values, np.array(sample.getPixels(), dtype=float), actual, predicted)
          #backprop(hidden_weights, output_weights, actual, predicted, sample, hidden_layer_values, learning_rate= 0.01)
      elif predicted < 0.5 and actual == 1:
          backprop(hidden_weights, output_weights, hidden_biases, output_bias, hidden_layer_values, np.array(sample.getPixels(), dtype=float), actual, predicted)
          #backprop(hidden_weights, output_weights, actual, predicted, sample, hidden_layer_values, learning_rate= 0.01)

  # now test the test data to see how accurate it is    
  data_test = loadDataFile('data/facedata/facedatavalidation', 301, 60, 70)
  labels_test = loadLabelsFile('data/facedata/facedatavalidationlabels', 301)
  
  accuracies = []

  for idx in range(301):

    sample = data_test[idx]
    label = labels_test[idx]
    total_sum = 0

    for node in range(1000):
        total_sum = 0
        for i in range(70):
          for j in range(60):
              total_sum += hidden_weights[node][i][j] * sample.getPixel(i, j)

        hidden_layer_values.append(sigmoid_activation(total_sum + hidden_biases[node]))    
        
        final_output = 0
    
        for i in range(len(hidden_layer_values)):
            final_output += output_weights[i] * hidden_layer_values[i]

        predicted = sigmoid_activation(final_output + output_bias)

        actual = label

        if (total_sum > 0.5 and label == 1):
          accuracies.append(1)
        elif (total_sum < 0.5 and label == 0):
          accuracies.append(1)
        else:
          accuracies.append(0)
  
  #print(accuracies)
  return np.mean(accuracies), np.std(accuracies)

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