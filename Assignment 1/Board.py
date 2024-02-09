import numpy
import random

class Board:


    def __init__(self):
        self.rows = 15
        self.cols = 15
        self.target = (random.randint(0, 14), random.randint(0, 14))
        self.initial = (random.randint(0, 14), random.randint(0, 14))
        self.openList = {}
        self.closedList = {}
        self.board = None

    def createBoard (self, rows, cols):
        
        unblocked_prob = .7

        self.board = numpy.uint8(numpy.random.uniform(size=(rows, cols)) > unblocked_prob)

        print(self.initial)

        print(self.board)
        return self.board
    
    def Astar(board):
        
        pass
    

board  = Board()
grid = board.createBoard(15, 15)

