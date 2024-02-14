import numpy
import random
import heapq
from functools import cmp_to_key

class Board:

# f = g + h (total cost of the path)
# g = inital state to current state
# h = current state to the goal state

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.target = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        self.initial = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        self.openList = []
        heapq.heapify(self.openList)
        self.closedList = set()
        self.board = None

    def createBoard (self):
        
        unblocked_prob = .7

        self.board = numpy.uint8(numpy.random.uniform(size=(self.rows, self.cols)) > unblocked_prob)

        print(self.initial)
        print(self.target)

        self.board[self.target[0]][self.target[1]] = 2
        self.board[self.initial[0]][self.initial[1]] = 3
        print(self.board)

        return self.board
    
    def calculateHeuristic(self, i, j):
        distance = abs(self.target[0] - i) + abs(self.target[1] - j)
        return distance
    
    

    # Negating g value  
    def Astar(self, grid):
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        path = []
        found_destination = False

        heapq.heappush(self.openList, (self.calculateHeuristic(self.initial[0], self.initial[1]), 0, self.calculateHeuristic(self.initial[0], self.initial[1]) * -1, self.initial[0],  self.initial[1]))
    
        while self.openList and not found_destination:
            fValue, gValue, _, row, col, = heapq.heappop(self.openList)

            
            for dr, dc in directions:
                self.closedList.add((row, col, fValue))
                r, c = row + dr, col + dc

                if r == self.target[0] and c == self.target[1]:
                    print(r, c)
                    print("Path haS BEEN FOUND")
                    found_destination = True
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.board[r][c] != 1 and (r, c, fValue) not in self.closedList:
                    print("Checking for path")
                    print(r, c)
                    hVal = self.calculateHeuristic(r, c)
                    fVal = hVal - gValue
                    heapq.heappush(self.openList, (fVal, gValue - 1, hVal, r, c))
                    path.append((r, c))
                    self.board[r][c] = 5


        print(self.board)
        #print(path)

board  = Board(5, 5)
grid = board.createBoard()
board.Astar(grid)
print(board.closedList)


