import numpy
import random
import heapq
from functools import cmp_to_key

class Astar:


   def __init__(self):
        self.rows = 15
        self.cols = 15
        self.target = (random.randint(0, 14), random.randint(0, 14))
        self.initial = (random.randint(0, 14), random.randint(0, 14))
        self.openList = []
        heapq.heapify(self.openList)
        self.closedList = set()
        self.board = None

   def createBoard (self, rows, cols):
        
        unblocked_prob = .7

        self.board = numpy.uint8(numpy.random.uniform(size=(rows, cols)) > unblocked_prob)

        print(f'Start Point: {self.initial}')
        print(f'End Point: {self.target}')

        self.board[self.initial[0]][self.initial[1]] = 4
        self.board[self.target[0]][self.target[1]] = 6
        
        
        print('Board Before Search:')
        print(self.board)

        return self.board
   
   def calculateHeuristic(self, currentRow, currentCol):
        goalRow = self.target[0]
        goalCol = self.target[1]
        manhattanDistance  = abs(currentRow - goalRow) + abs(goalCol - currentCol)
        return manhattanDistance

   def aStar(self, grid):
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        # distance to the current state (in the beginning it's 0 since no moves have been made)
        g_val = 0

        # f = g + h (the distance to the current state plus the heuristic function)
        f_val = 0

        # f_val, g_val, h_val, row, column
        heapq.heappush(self.openList, (f_val, g_val, self.calculateHeuristic(self.initial[0], self.initial[1]), self.initial[0], self.initial[1]))

        while self.openList:
             
             f_val, g_val, h_val, currRow, currCol = heapq.heappop(self.openList)

             if currRow == self.target[0] and currCol == self.target[1]:
                  print('Goal has been found!')
                  break

             for dr, dc in directions:
                  
                  self.closedList.add((currRow, currCol, f_val))

                  row = currRow + dr
                  col = currCol + dc

                  if row in range(self.rows) and col in range(self.cols) and self.board[row][col] == '0' and self.board[row][col] not in self.closedList:
                      print('made it')
                      h = self.calculateHeuristic(row, col)
                      f = h + g_val
                      heapq.heappush(self.openList, (f, g_val, h, row, col))
                      self.board[row][col] = "5"
                      g_val += 1

        print('After Board Search')
        print(self.board)

board  = Astar()
grid = board.createBoard(15, 15)
board.aStar(grid)