# samples.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import numpy as np
import zipfile
import os

## Constants
DATUM_WIDTH_FACE = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

class Perceptron:
 
  def __init__(self, data,width,height):
    """
    Create a new datum from file input (standard MNIST encoding).
    """
    DATUM_HEIGHT = height
    DATUM_WIDTH=width
    self.height = DATUM_HEIGHT
    self.width = DATUM_WIDTH
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

# function used to train the data and get the final weights which will be used on the actual data  
def train_face(n):

  epochs = 10
  data = loadDataFile('data/facedata/facedatatrain', 451, 60, 70)
  labels = loadLabelsFile('data/facedata/facedatatrainlabels', 451)
  weights = np.random.uniform(low=-1e9, high=1e9, size=(70, 60))
  num_samples = int(n * 451)

  accuracies = []
  
   
  for epoch in range(epochs):
    num_accurate = 0
    for _ in range(num_samples):

      idx = np.random.randint(0, 451)
      sample = data[idx]
      
      total_sum = 0
  

      for i in range(70):
        for j in range(60):
          total_sum += sample.getPixel(i, j) * weights[i][j] 

      label = labels[idx]


      if total_sum > 0 and label == 0:
        for i in range(70):
          for j in range(60):
            weights[i][j] += weights[i][j] - sample.getPixel(i, j) 
        
      # Predicted output is not a face, but actual output is a face
      elif total_sum < 0 and label == 1:
        for i in range(70):
          for j in range(60):
            weights[i][j] += weights[i][j] + sample.getPixel(i, j) 
      else:
        num_accurate += 1
    
    percentage_accurate = num_accurate / num_samples
    accuracies.append(percentage_accurate)        

  return np.average(accuracies), np.std(accuracies)

# check the accuracy of the model after the training
def test_model():
  pass


# Testing
def _test():
  average, std = train_face(0.8)
  print(average, std)
if __name__ == "__main__":
  _test()  
