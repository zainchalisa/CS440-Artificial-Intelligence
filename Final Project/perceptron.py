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

# function used to train the data and get the final weights which will be used on the actual data  
def train_face(n):

  epochs = 10
  data = loadDataFile('data/facedata/facedatatrain', 451, 60, 70)
  labels = loadLabelsFile('data/facedata/facedatatrainlabels', 451)
  weights = np.random.randint(low= -1e9, high= 1e9, size=(70, 60))
  num_samples = int(n * 451)

  accuracies = []

  bias = 1
   
  for epoch in range(epochs):
    for _ in range(num_samples):

      idx = np.random.randint(0, 451)
      sample = data[idx]
      
      total_sum = bias

      for i in range(70):
        for j in range(60):
          total_sum += sample.getPixel(i, j) * weights[i][j] 

      label = labels[idx]

      if total_sum > 0 and label == 0:
        bias -= 1
        for i in range(70):
          for j in range(60):
            weights[i][j] += weights[i][j] - sample.getPixel(i, j) 
        
      # Predicted output is not a face, but actual output is a face
      elif total_sum < 0 and label == 1:
        bias += 1
        for i in range(70):
          for j in range(60):
            weights[i][j] += weights[i][j] + sample.getPixel(i, j) 

  ################### END OF TRAINING MODEL CODE ###################            
      
  # now test the test data to see how accurate it is    
  data_test = loadDataFile('data/facedata/facedatatest', 150, 60, 70)
  labels_test = loadLabelsFile('data/facedata/facedatatestlabels', 150)

  for idx in range(150):

    sample = data_test[idx]
    label = labels_test[idx]
    total_sum = 0


    for i in range(70):
      for j in range(60):
        total_sum += sample.getPixel(i, j) * weights[i][j] 
        #print(f'Pixel{sample.getPixel(i, j)}, Weight{weights[i][j]}')

    total_sum += bias

    #print(total_sum)
    if (total_sum > 0 and label == 1) or (total_sum < 0 and label == 0):
      #print('here')
      accuracies.append(1)
    else:
      accuracies.append(0)
  
  #print(accuracies)
  return np.mean(accuracies), np.std(accuracies)

def train_digit(n):

  epochs = 20
  data = loadDataFile('data/digitdata/trainingimages', 5000, 28, 28)
  labels = loadLabelsFile('data/digitdata/traininglabels', 5000)
  weights = np.random.uniform(low=-1e9, high=1e9, size=(10, 28, 28))
  num_samples = int(n * 5000)

  accuracies = []

  bias = np.random.uniform(low=-1e9, high=1e9, size=10)

  for epoch in range(epochs):
    num_accurate = 0
    for _ in range(num_samples):
      
      # random idx from the training data
      idx = np.random.randint(0, 5000)

      # the image at that index in the training data
      image = data[idx]

      predicted_digit = 0
      max_sum = 0
  
      # the loop which we'll use to find out the predicited digit (this digit is the one with the highest total_sum)
      for digit in range(0, 9):
        total_sum = 0 
        for i in range(28):
          for j in range(28):
            total_sum += image.getPixel(i, j) * weights[digit][i][j]
        
        # adds the bias value associated to the current digit to the total sum
        total_sum += bias[digit]

        # checks if we need to update the max_sum and predicted_digit
        if total_sum > max_sum:
          max_sum = total_sum
          predicted_digit = digit

      # gets the actual digit which the image represents 
      real_digit = labels[idx]

      # checks to see if our prediction is accurate or not, if not we will update the weights and bias
      if predicted_digit != real_digit:
        bias[predicted_digit] -= 1
        bias[real_digit] += 1
        for i in range(28):
          for j in range(28):
            weights[predicted_digit][i][j] = weights[predicted_digit][i][j] - image.getPixel(i, j)
            weights[real_digit][i][j] =  weights[real_digit][i][j] + image.getPixel(i, j)
        #print('weights updated')
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
  average, std = train_face(.8)
  print(average, std)

if __name__ == "__main__":
  _test()  
