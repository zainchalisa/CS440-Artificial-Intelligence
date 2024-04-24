import util
import numpy as np
import zipfile
import os

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
        items.append(Perceptron(data, DATUM_WIDTH, DATUM_HEIGHT))
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
  

def nn_face(n):
  
  epochs = 10
  data = loadDataFile('data/facedata/facedatatrain', 451, 60, 70)
  labels = loadLabelsFile('data/facedata/facedatatrainlabels', 451)
  weights = np.random.randint(low= -400, high= 400, size=(1000, 70, 60)) # we need different weights for each of the nodes on the hidden layer
  biases = np.random.randint(low=-200, high=200, size=1000)
  hidden_layer = [] 
  num_samples = int(n * 451)

    # when creating the neural network we need the following:
        # the input layer (this layer will be the individual pixels of the image)
        # the hidden layer (this layer will use the sigmoid function g(Z) = (1/(1 + e^-2)))
        # the output layer (this layer is computed using sigmoid of each of the values computed in the hidden layer)

  for epoch in range(epochs):
    for _ in num_samples:
      
      idx = np.random.randint(0, 451)
      sample = data[idx]
      
      
        




def nn_digit(n):
  
  epochs = 10
  data = loadDataFile('data/digitdata/trainingimages', 5000, 28, 28)
  labels = loadLabelsFile('data/digitdata/traininglabels', 5000)
  weights = np.random.randint(low=-400, high=400, size=(10, 28, 28))
  num_samples = int(n * 5000)



def _test():
  average, std = nn_digit(.5)
  print(average, std)

if __name__ == "__main__":
  _test()  