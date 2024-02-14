import numpy
import random
import heapq

class Board:
    def __init__(self, rows, cols):
        self.parent_dict = {}
        self.rows = rows
        self.cols = cols
        self.target = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        self.initial = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        self.openList = []
        heapq.heapify(self.openList)
        self.closedList = set()
        self.board = None

    def createBoard(self):
        unblocked_prob = 0.7
        self.board = numpy.uint8(numpy.random.uniform(size=(self.rows, self.cols)) > unblocked_prob)

        print("Initial:", self.initial)
        print("Target:", self.target)

        self.board[self.target[0]][self.target[1]] = 2
        self.board[self.initial[0]][self.initial[1]] = 3
        print(self.board)

        return self.board

    def calculateHeuristic(self, i, j):
        distance = abs(self.target[0] - i) + abs(self.target[1] - j)
        return distance

    def ForwardAStar_WithBiggerG(self, grid):
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        found_destination = False

        heapq.heappush(self.openList, (self.calculateHeuristic(self.initial[0], self.initial[1]), 0, self.calculateHeuristic(self.initial[0], self.initial[1]), self.initial[0], self.initial[1]))

        while self.openList and not found_destination:
            print(self.openList)
            fValue, gValue, _, row, col, = heapq.heappop(self.openList)
            self.closedList.add((row, col))
            print(fValue, gValue)

            for dr, dc in directions:
                r, c = row + dr, col + dc

                if r == self.target[0] and c == self.target[1]:
                    print(r, c)
                    print("Path has been found")
                    found_destination = True
                    self.parent_dict[(r, c)] = ((row, col), gValue)
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.board[r][c] != 1 and (r, c) not in self.closedList:
                    hVal = self.calculateHeuristic(r, c)
                    fVal = hVal - gValue
                    heapq.heappush(self.openList, (fVal, gValue + 1, hVal, r, c))
                    self.parent_dict[(r, c)] = ((row, col), gValue + 1)

        print(self.board)
        print(self.parent_dict)

        if found_destination:
            path = [(self.target[0], self.target[1])]
            current_cell = (self.target[0], self.target[1])

            while current_cell != self.initial:
                current_cell, _ = self.parent_dict[current_cell]
                path.append(current_cell)

            path.reverse()
            print("Optimal path:", path)
        else:
            print("No path found.")
        

    def ForwardAStar_WithSmallerG(self, grid):
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        found_destination = False

        heapq.heappush(self.openList, (self.calculateHeuristic(self.initial[0], self.initial[1]), 0, self.calculateHeuristic(self.initial[0], self.initial[1]) * -1, self.initial[0], self.initial[1]))

        while self.openList and not found_destination:
            print(self.openList)
            fValue, gValue, _, row, col, = heapq.heappop(self.openList)
            self.closedList.add((row, col))
            print(fValue, gValue)

            for dr, dc in directions:
                r, c = row + dr, col + dc

                if r == self.target[0] and c == self.target[1]:
                    print(r, c)
                    print("Path has been found")
                    found_destination = True
                    self.parent_dict[(r, c)] = ((row, col), gValue)
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.board[r][c] != 1 and (r, c) not in self.closedList:
                    hVal = self.calculateHeuristic(r, c)
                    fVal = hVal - gValue
                    heapq.heappush(self.openList, (fVal, gValue - 1, hVal, r, c))
                    self.parent_dict[(r, c)] = ((row, col), gValue - 1)

        print(self.board)
        print(self.parent_dict)

        if found_destination:
            path = [(self.target[0], self.target[1])]
            current_cell = (self.target[0], self.target[1])

            while current_cell != self.initial:
                current_cell, _ = self.parent_dict[current_cell]
                path.append(current_cell)

            path.reverse()
            print("Optimal path:", path)
        else:
            print("No path found.")

board = Board(10, 10)
grid = board.createBoard()
board.ForwardAStar_WithSmallerG(grid)
