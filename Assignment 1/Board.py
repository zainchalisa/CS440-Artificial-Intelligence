import numpy
import random
import heapq
from functools import cmp_to_key

class Board:

# f = g + h (total cost of the path)
# g = inital state to current state
# h = current state to the goal state

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

        print(self.initial)
        print(self.target)

        self.board[self.target[0]][self.target[1]] = 2
        self.board[self.initial[0]][self.initial[1]] = 3

        return self.board
    
    def calculateHeuristic(self, i, j):
        distance = abs(self.target[0] - i) + abs(self.target[1] - j)
        return distance
    
    def priority(node):
        fValue, row, col, gValue, hValue = node
        return(fValue, -gValue)
    
    def Astar(self, grid):
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        
        
        heapq.heappush(self.openList, (self.calculateHeuristic(self.initial[0], self.initial[1]), self.initial[0],  self.initial[1], 0, self.calculateHeuristic(self.initial[0], self.initial[1])))
        d = 1
        
        while self.openList:
            fValue, row, col, gValue, hValue = heapq.heappop(self.openList)

            if row == self.target[0] and col == self.target[1]:
                print("Path haS BEEN FOUND")
                break
                

            
            for dr, dc in directions:
                self.closedList.add((row, col, fValue))
                r, c = row + dr, col + dc
                if r in range(self.rows) and c in range(self.cols) and self.board[r][c] == 0 and self.board[r][c] not in self.closedList:
                    hVal = self.calculateHeuristic(r, c)
                    fVal = hVal + d
                    heapq.heappush(self.openList, (fVal, r, c, d, hVal))
                    self.board[r][c] = "5"
                    d += 1

            

        print(self.board)
        



       
    

board  = Board()
grid = board.createBoard(15, 15)
board.Astar(grid)
print(board.closedList)


