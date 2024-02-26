import numpy
import random
import heapq
import pygame
import time

class Board:
    def __init__(self, rows, cols):
        self.parent_dict = {}
        self.final_path = []
        self.expanded_nodes = 0
        self.found = False
        self.rows = rows
        self.cols = cols
        self.target = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        self.initial = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        self.openList = []
        self.h_matrix = numpy.zeros((self.rows, self.cols))
        heapq.heapify(self.openList)
        self.closedList = {}
        self.board = None
        self.planning_board = None
        self.agent = [self.initial[0], self.initial[1]]

        pygame.init()
        self.screen_width = 600
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("A* Pathfinding Visualization")
        self.clock = pygame.time.Clock()


    def createBoard(self):
        unblocked_prob = 0.7
        self.board = numpy.uint8(numpy.random.uniform(size=(self.rows, self.cols)) > unblocked_prob)
        self.planning_board = numpy.zeros((self.rows, self.cols))
        
        # populating heuristic matrix 
        for i in range(self.rows):
            for j in range(self.cols):
                self.h_matrix[i][j] = self.calculateHeuristic(i, j)

    
        print("Initial:", self.initial)
        print("Target:", self.target)

        self.board[self.target[0]][self.target[1]] = 2
        self.board[self.initial[0]][self.initial[1]] = 3

        print('Initial Board')
        print(self.board)

    def calculateHeuristic(self, i, j):
        distance = abs(self.target[0] - i) + abs(self.target[1] - j)
        return distance
    
    def draw_grid(self, path, wait_for_click = False):
        
        cell_width = self.screen_width // self.cols
        cell_height = self.screen_height // self.rows

        for i in range(self.rows):
            for j in range(self.cols):
                color = (255, 255, 255) if self.board[i][j] == 0 else (0, 0, 0)  # White for unblocked, black for blocked
                pygame.draw.rect(self.screen, color, (j * cell_width, i * cell_height, cell_width, cell_height))

        if path != None:
            for i in range(len(path)):
               color = (255, 165, 0)
               pygame.draw.rect(self.screen, color, (path[i][1] * cell_width, path[i][0] * cell_height, cell_width, cell_height))

        # Draw initial position
        pygame.draw.rect(self.screen, (0, 255, 0), (self.initial[1] * cell_width, self.initial[0] * cell_height, cell_width, cell_height))
        # Draw target position
        pygame.draw.rect(self.screen, (255, 0, 0), (self.target[1] * cell_width, self.target[0] * cell_height, cell_width, cell_height))
        
        if wait_for_click:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pygame.display.flip()
                        return  # Exit the function if the mouse is clicked
                    
        

    def show_popup(self, message):
        popup_width = 300
        popup_height = 100
        popup_screen = pygame.display.set_mode((popup_width, popup_height))
        pygame.display.set_caption("Popup Message")

        font = pygame.font.Font(None, 36)
        text = font.render(message, True, (0, 0, 0))
        text_rect = text.get_rect(center=(popup_width // 2, popup_height // 2))

        popup_screen.fill((255, 255, 255))
        popup_screen.blit(text, text_rect)

        pygame.display.flip()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()
    
    def ForwardAStar_WithSmallerG(self) -> list:
        #print(self.openList)
        self.openList = []
        heapq.heapify(self.openList)
        found_destination = False
        if self.agent[0] != self.initial[0] or self.agent[1] != self.initial[1]: 
            h_val = self.h_matrix[self.agent[0]][self.agent[1]]
            g_val = self.closedList[(self.agent[0], self.agent[1])]
            f_val = h_val + g_val 
            parentRow, parentCol = self.parent_dict[(self.agent[0], self.agent[1])]
            if (f_val, g_val, h_val, self.agent[0], self.agent[1], parentRow, parentCol) not in self.openList:
                heapq.heappush(self.openList, (f_val, g_val, h_val, self.agent[0], self.agent[1], parentRow, parentCol))
        
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        self.closedList = {}

        # First execution of planning
        if self.agent[0] == self.initial[0] and self.agent[1] == self.initial[1]: 
            heapq.heappush(self.openList, (self.h_matrix[self.agent[0]][self.agent[1]], 0, self.calculateHeuristic(self.initial[0], self.initial[1]), self.initial[0], self.initial[1], -1, -1))
        
        # Agent is somewhere else besides initial cell
        

        while self.openList and not found_destination:
            #print(f'Open List: {self.openList}')
            fValue, gValue, _, childRow, childCol, parentRow, parentCol = heapq.heappop(self.openList)
            self.closedList[(childRow, childCol)] = gValue
            self.parent_dict[(childRow, childCol)] = (parentRow, parentCol)

            #print(f'Heap: F Value: {fValue},  G Value: {gValue}, Row: {childRow}, Col: {childCol}')
            #print(f'Parent Dictionary: {self.parent_dict}')
            for dr, dc in directions:
                r, c = childRow + dr, childCol + dc

                if r == self.target[0] and c == self.target[1]:
                    #print(r, c)
                    print("Path has been found")
                    found_destination = True
                    self.parent_dict[(r, c)] = (childRow, childCol)
                    self.closedList[(r, c)] = gValue + 1
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.planning_board[r][c] != 1 and (r, c) not in self.closedList:
                    hVal = self.h_matrix[r][c]
                    fVal = hVal + (gValue + 1)
                    if (fVal, gValue + 1, hVal, r, c, childRow, childCol) not in self.openList:
                        heapq.heappush(self.openList, (fVal, gValue + 1, hVal, r, c, childRow, childCol))
                    

        #print(self.planning_board)
        #print(self.closedList)

        if found_destination:
            self.path = [(self.target[0], self.target[1])]
            current_cell = (self.target[0], self.target[1])
            #print(f"Parent Dict: {self.parent_dict}")
            #print(f'Agent: {self.agent}')
            while current_cell != tuple(self.agent):
                #print(f'Current Cell: {current_cell}')
                current_cell = self.parent_dict[current_cell]
                self.path.append(current_cell)

            self.path.reverse()
            #print(f'Before Execution: {self.path}')
            self.expanded_nodes += len(self.closedList.keys())
            self.execution_S(self.path)
            
        else:
            print('no path found')
            return None


    def ForwardAStar_WithBiggerG(self) -> list:
        self.openList = []
        heapq.heapify(self.openList)
        found_destination = False
        if self.agent[0] != self.initial[0] or self.agent[1] != self.initial[1]: 
            h_val = self.h_matrix[self.agent[0]][self.agent[1]]
            g_val = self.closedList[(self.agent[0], self.agent[1])]
            f_val = h_val - g_val 
            parentRow, parentCol = self.parent_dict[(self.agent[0], self.agent[1])]
            heapq.heappush(self.openList, (f_val, g_val, h_val, self.agent[0], self.agent[1], parentRow, parentCol))
        
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        self.closedList = {}

        # First execution of planning
        if self.agent[0] == self.initial[0] and self.agent[1] == self.initial[1]: 
            heapq.heappush(self.openList, (self.h_matrix[self.agent[0]][self.agent[1]], 0, self.calculateHeuristic(self.initial[0], self.initial[1]), self.initial[0], self.initial[1], -1, -1))
        
        # Agent is somewhere else besides initial cell
        

        while self.openList and not found_destination:
            #print(f'Open List: {self.openList}')
            fValue, gValue, _, childRow, childCol, parentRow, parentCol = heapq.heappop(self.openList)
            self.closedList[(childRow, childCol)] = gValue
            self.parent_dict[(childRow, childCol)] = (parentRow, parentCol)

            #print(f'Heap: F Value: {fValue},  G Value: {gValue}, Row: {childRow}, Col: {childCol}')
            #print(f'Parent Dictionary: {self.parent_dict}')
            for dr, dc in directions:
                r, c = childRow + dr, childCol + dc

                if r == self.target[0] and c == self.target[1]:
                    #print(r, c)
                    print("Path has been found")
                    found_destination = True
                    self.parent_dict[(r, c)] = (childRow, childCol)
                    self.closedList[(r, c)] = gValue - 1
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.planning_board[r][c] != 1 and (r, c) not in self.closedList:
                    hVal = self.h_matrix[r][c]
                    fVal = hVal - (gValue - 1)
                    heapq.heappush(self.openList, (fVal, gValue - 1, hVal, r, c, childRow, childCol))
                    

        #print(self.planning_board)
        #print(self.closedList)

        if found_destination:
            self.path = [(self.target[0], self.target[1])]
            current_cell = (self.target[0], self.target[1])
            #print(f"Parent Dict: {self.parent_dict}")
            #print(f'Agent: {self.agent}')
            while current_cell != tuple(self.agent):
                #print(f'Current Cell: {current_cell}')
                current_cell = self.parent_dict[current_cell]
                self.path.append(current_cell)

            self.path.reverse()
            #print(f'Before Execution: {self.path}')
            self.expanded_nodes += len(self.closedList.keys())
            self.execution_B(self.path)
            
        else:
            print('no path found')
            return None

    def reset_board(self):
        self.openList = []
        heapq.heapify(self.openList)
        self.closedList = {}
        self.parent_dict = {}
        self.path = []
        self.final_path = []
        self.planning_board = numpy.zeros((self.rows, self.cols))
        self.agent = [self.initial[0], self.initial[1]]
        self.expanded_nodes = 0


    def update_heuristic(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in self.closedList:
                   # print("updating heuristic value")
                    g_val = self.closedList[(i, j)]
                    initial_val = self.closedList[tuple(self.target)]
                    #print(initial_val)
                    #print(f'H-Val Before: {self.h_matrix[i][j]}')
                    new_h = (initial_val * -1) - (-1 * g_val )
                    self.h_matrix[i][j] = new_h
                    #print(f'H-Val After: {self.h_matrix[i][j]}')
                    #print(self.h_matrix)
                else:
                    continue
        #print(self.h_matrix)
    
    def backwards(self):
        self.openList = []
        heapq.heapify(self.openList)
        found_destination = False
        if self.agent[0] != self.initial[0] or self.agent[1] != self.initial[1]: 
            h_val = self.h_matrix[self.target[0]][self.target[1]]
            g_val = self.closedList[(self.target[0], self.target[1])]
            f_val = h_val - g_val 
            self.parent_dict[(self.target[0], self.target[1])] = (-1, -1)
            parentRow, parentCol = self.parent_dict[(self.target[0], self.target[1])]
            heapq.heappush(self.openList, (f_val, g_val, h_val, self.target[0], self.target[1], parentRow, parentCol))
        
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        self.closedList = {}

        # First execution of planning
        if self.agent[0] == self.initial[0] and self.agent[1] == self.initial[1]: 
            heapq.heappush(self.openList, (self.h_matrix[self.target[0]][self.target[1]], 0, self.calculateHeuristic(self.target[0], self.target[1]), self.target[0], self.target[1], -1, -1))
        
        # Agent is somewhere else besides initial cell
        

        while self.openList and not found_destination:
            #print(f'Open List: {self.openList}')
            fValue, gValue, _, childRow, childCol, parentRow, parentCol = heapq.heappop(self.openList)
            self.closedList[(childRow, childCol)] = gValue
            self.parent_dict[(childRow, childCol)] = (parentRow, parentCol)

            #print(f'Heap: F Value: {fValue},  G Value: {gValue}, Row: {childRow}, Col: {childCol}')
            #print(f'Parent Dictionary: {self.parent_dict}')
            for dr, dc in directions:
                r, c = childRow + dr, childCol + dc

                if r == self.agent[0] and c == self.agent[1]:
                    #print(r, c)
                    print("Path has been found")
                    found_destination = True
                    self.parent_dict[(r, c)] = (childRow, childCol)
                    self.closedList[(r, c)] = gValue - 1
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.planning_board[r][c] != 1 and (r, c) not in self.closedList:
                    hVal = self.h_matrix[r][c]
                    fVal = hVal - (gValue - 1)
                    if (fVal, gValue - 1, hVal, r, c, childRow, childCol) not in self.openList:
                        heapq.heappush(self.openList, (fVal, gValue - 1, hVal, r, c, childRow, childCol))
                    

        #print(self.planning_board)
        #print(self.closedList)

        if found_destination:
            self.path = [(self.agent[0], self.agent[1])]
            current_cell = (self.agent[0], self.agent[1])
            #print(f"Parent Dict: {self.parent_dict}")
            #print(f'Agent: {self.agent}')
            while current_cell != self.target:
                #print(f'Current Cell: {current_cell}')
                current_cell = self.parent_dict[current_cell]
                self.path.append(current_cell)

            
            #print(f'Before Execution: {self.path}')
            self.expanded_nodes += len(self.closedList.keys())
            self.execution_backwards(self.path)
            
        else:
            print('no path found')
            return None
        
    def BackwardAStar_WithBiggerG(self) -> list:
        self.openList = []
        heapq.heapify(self.openList)
        found_destination = False
        if self.agent[0] != self.initial[0] or self.agent[1] != self.initial[1]: 
            h_val = self.h_matrix[self.target[0]][self.target[1]]
            g_val = self.closedList[(self.target[0], self.target[1])]
            f_val = h_val - g_val 
            #print(self.agent)
            #print(self.parent_dict)
            self.parent_dict[(self.target[0], self.target[1])] = (-1, -1)
            parentRow, parentCol = self.parent_dict[(self.target[0], self.target[1])]
            if (f_val, g_val, h_val, self.agent[0], self.agent[1], parentRow, parentCol) not in self.openList:
                heapq.heappush(self.openList, (f_val, g_val, h_val, self.agent[0], self.agent[1], parentRow, parentCol))
        
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        self.closedList = {}

        # First execution of planning
        if self.agent[0] == self.initial[0] and self.agent[1] == self.initial[1]: 
            heapq.heappush(self.openList, (self.h_matrix[self.target[0]][self.target[1]], 0, self.h_matrix[self.target[0]][self.target[1]], self.target[0], self.target[1], -1, -1))
        
        # Agent is somewhere else besides initial cell
        

        while self.openList and not found_destination:
            print(f'Open List: {self.openList}')
            fValue, gValue, _, childRow, childCol, parentRow, parentCol = heapq.heappop(self.openList)
            self.closedList[(childRow, childCol)] = gValue
            self.parent_dict[(childRow, childCol)] = (parentRow, parentCol)

            print(f'Heap: F Value: {fValue},  G Value: {gValue}, Row: {childRow}, Col: {childCol}')
            #print(f'Parent Dictionary: {self.parent_dict}')
            for dr, dc in directions:
                r, c = childRow + dr, childCol + dc

                if r == self.agent[0] and c == self.agent[1]:
                    #print(r, c)
                    #print("Path has been found")
                    found_destination = True
                    self.parent_dict[(r, c)] = (childRow, childCol)
                    self.closedList[(r, c)] = gValue - 1
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.planning_board[r][c] != 1 and (r, c) not in self.closedList:
                    hVal = self.h_matrix[r][c]
                    fVal = hVal - (gValue - 1)
                    heapq.heappush(self.openList, (fVal, gValue - 1, hVal, r, c, childRow, childCol))
                    

        #print(self.planning_board)
        #print(self.closedList)

        if found_destination:
            self.path = [(self.agent[0], self.agent[1])]
            current_cell = (self.agent[0], self.agent[1])
            #print(f"Parent Dict: {self.parent_dict}")
            #print(f'Agent: {self.agent}')
            while current_cell != self.target:
                current_cell = self.parent_dict[current_cell]
                #print(f'Current Cell: {current_cell}')
                self.path.append(current_cell)
            
            #print(self.path)
            #print(f'Before Execution: {self.path}')
            self.expanded_nodes += len(self.closedList.keys())
            self.execution_backwards(self.path)
            
        else:
            print('no path found')
            return None

    def AdaptiveAStar_WithBiggerG(self) -> list:
        #print(self.h_matrix)
        self.openList = []
        heapq.heapify(self.openList)
        found_destination = False
        if self.agent[0] != self.initial[0] or self.agent[1] != self.initial[1]: 
            h_val = self.h_matrix[self.agent[0]][self.agent[1]]
            g_val = self.closedList[(self.agent[0], self.agent[1])]
            f_val = h_val - g_val 
            parentRow, parentCol = self.parent_dict[(self.agent[0], self.agent[1])]
            heapq.heappush(self.openList, (f_val, g_val, h_val, self.agent[0], self.agent[1], parentRow, parentCol))
        
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        self.closedList = {}

        # First execution of planning
        if self.agent[0] == self.initial[0] and self.agent[1] == self.initial[1]: 
            heapq.heappush(self.openList, (self.h_matrix[self.agent[0]][self.agent[1]], 0, self.h_matrix[self.agent[0]][self.agent[1]], self.initial[0], self.initial[1], -1, -1))
        
        # Agent is somewhere else besides initial cell
        

        while self.openList and not found_destination:
            #print(f'Open List: {self.openList}')
            fValue, gValue, _, childRow, childCol, parentRow, parentCol = heapq.heappop(self.openList)
            self.closedList[(childRow, childCol)] = gValue
            self.parent_dict[(childRow, childCol)] = (parentRow, parentCol)

            #print(f'Heap: F Value: {fValue},  G Value: {gValue}, Row: {childRow}, Col: {childCol}')
            #print(f'Parent Dictionary: {self.parent_dict}')
            for dr, dc in directions:
                r, c = childRow + dr, childCol + dc

                if r == self.target[0] and c == self.target[1]:
                    #print(r, c)
                    #print("Path has been found")
                    found_destination = True
                    self.parent_dict[(r, c)] = (childRow, childCol)
                    self.closedList[(r, c)] = gValue - 1
                    break

                elif r in range(self.rows) and c in range(self.cols) and self.planning_board[r][c] != 1 and (r, c) not in self.closedList:
                    hVal = self.h_matrix[r][c]
                    fVal = hVal - (gValue - 1)
                    heapq.heappush(self.openList, (fVal, gValue - 1, hVal, r, c, childRow, childCol))
                    

        #print(self.planning_board)
        #print(self.closedList)

        if found_destination:
            self.path = [(self.target[0], self.target[1])]
            current_cell = (self.target[0], self.target[1])
            #print(f"Parent Dict: {self.parent_dict}")
            #print(f'Agent: {self.agent}')
            while current_cell != tuple(self.agent):
                #print(f'Current Cell: {current_cell}')
                current_cell = self.parent_dict[current_cell]
                self.path.append(current_cell)

            self.path.reverse()
            #print(self.closedList)
            #print(self.h_matrix)
            self.update_heuristic()
            #print(self.h_matrix)
            self.expanded_nodes += len(self.closedList.keys())
            self.execution_A(self.path)
            
        else:
            print('no path found')
            return None

    def execution_B(self, path):
        #print(f'Print Execution \n {self.h_matrix}')
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        #print(f"Iterative Path {path}")
        for (r, c) in self.path:
            #print(self.agent)
            #print((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if nr in range(self.rows) and nc in range(self.cols) and self.board[nr][nc] == 1:
                    self.planning_board[nr][nc] = 1

            if tuple((r,c)) == self.target:
                print("Path has been found!")
                #print(self.planning_board)
                self.found = True
                return self.final_path
                
            
            if self.board[r][c] == 1:
                #print(f'Blocked Cell at RC: {[r, c]}')
                self.planning_board[r][c] = 1
                #print(f"Parent Dict Before: {self.parent_dict}")
                #print(f"Closed List: {self.closedList}")
                #print(f'Final Path List{self.final_path}')
                current_cell = (self.target[0], self.target[1])
                while current_cell != tuple(self.agent):
                    #print("Trying to find a better path.")
                    #print(f'Cell to be popped {current_cell}')
                    cell_to_be_popped = current_cell
                    current_cell = self.parent_dict[current_cell]
                    self.parent_dict.pop(cell_to_be_popped)
                    curr_row, curr_col = cell_to_be_popped
                    #print(cell_to_be_popped)
                    #print(self.final_path)
                    #self.final_path.pop(cell_to_be_popped)
                    #self.closedList.pop(cell_to_be_popped)

                #print(f"Parent Dict After: {self.parent_dict}")
                #print(self.agent)
                self.planning_board[self.agent[0]][self.agent[1]] = 5
                #print(f'Planning Grid:\n {self.planning_board}')
                #print(f'Execution Grid:\n {self.board}')
                #print(f"Path Before {self.path}")
                self.draw_grid(self.final_path, wait_for_click=True)
                self.path = self.ForwardAStar_WithBiggerG()
                #print(f"Path totototo {self.path}")
                if self.path is None:
                    #print('Path is cooked.')
                    return 
                break
            self.agent = tuple((r, c))   
            self.final_path.append((r, c))
            #print(f"Path {path}")
            #print(f"Agent is at yooyoyo {self.agent}")
    
    def execution_S(self, path):
        #print(f'Print Execution \n {self.h_matrix}')
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        #print(f"Iterative Path {path}")
        for (r, c) in self.path:
            #print(self.agent)
            #print((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if nr in range(self.rows) and nc in range(self.cols) and self.board[nr][nc] == 1:
                    self.planning_board[nr][nc] = 1

            if tuple((r,c)) == self.target:
                print("Path has been found!")
                #print(self.planning_board)
                self.found = True
                return self.final_path
                
            
            if self.board[r][c] == 1:
                #print(f'Blocked Cell at RC: {[r, c]}')
                self.planning_board[r][c] = 1
                #print(f"Parent Dict Before: {self.parent_dict}")
                #print(f"Closed List: {self.closedList}")
                #print(f'Final Path List{self.final_path}')
                current_cell = (self.target[0], self.target[1])
                while current_cell != tuple(self.agent):
                    #print("Trying to find a better path.")
                    #print(f'Cell to be popped {current_cell}')
                    cell_to_be_popped = current_cell
                    current_cell = self.parent_dict[current_cell]
                    self.parent_dict.pop(cell_to_be_popped)
                    curr_row, curr_col = cell_to_be_popped
                    #print(cell_to_be_popped)
                    #print(self.final_path)
                    #self.final_path.pop(cell_to_be_popped)
                    #self.closedList.pop(cell_to_be_popped)

                #print(f"Parent Dict After: {self.parent_dict}")
                #print(self.agent)
                self.planning_board[self.agent[0]][self.agent[1]] = 5
                #print(f'Planning Grid:\n {self.planning_board}')
                #print(f'Execution Grid:\n {self.board}')
                #print(f"Path Before {self.path}")
                self.draw_grid(self.final_path, wait_for_click=True)
                self.path = self.ForwardAStar_WithSmallerG()
                #print(f"Path totototo {self.path}")
                if self.path is None:
                    #print('Path is cooked.')
                    return 
                break
            self.agent = tuple((r, c))   
            self.final_path.append((r, c))
            #print(f"Path {path}")
            #print(f"Agent is at yooyoyo {self.agent}")

    def execution_A(self, path):
        #print(f'Print Execution \n {self.h_matrix}')
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        #print(f"Iterative Path {path}")
        for (r, c) in self.path:
            #print(self.agent)
            #print((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if nr in range(self.rows) and nc in range(self.cols) and self.board[nr][nc] == 1:
                    self.planning_board[nr][nc] = 1

            if tuple((r,c)) == self.target:
                print("Path has been found!")
                #print(self.planning_board)
                self.found = True
                return self.final_path
                
            
            if self.board[r][c] == 1:
                #print(f'Blocked Cell at RC: {[r, c]}')
                self.planning_board[r][c] = 1
                #print(f"Parent Dict Before: {self.parent_dict}")
                #print(f"Closed List: {self.closedList}")
                #print(f'Final Path List{self.final_path}')
                current_cell = (self.target[0], self.target[1])
                while current_cell != tuple(self.agent):
                    #print("Trying to find a better path.")
                    #print(f'Cell to be popped {current_cell}')
                    cell_to_be_popped = current_cell
                    current_cell = self.parent_dict[current_cell]
                    self.parent_dict.pop(cell_to_be_popped)
                    curr_row, curr_col = cell_to_be_popped
                    #print(cell_to_be_popped)
                    #print(self.final_path)
                    #self.final_path.pop(cell_to_be_popped)
                    #self.closedList.pop(cell_to_be_popped)

                #print(f"Parent Dict After: {self.parent_dict}")
                #print(self.agent)
                self.planning_board[self.agent[0]][self.agent[1]] = 5
                #print(f'Planning Grid:\n {self.planning_board}')
                #print(f'Execution Grid:\n {self.board}')
                #print(f"Path Before {self.path}")
                self.draw_grid(self.final_path, wait_for_click=True)
                self.path = self.AdaptiveAStar_WithBiggerG()
                #print(f"Path totototo {self.path}")
                if self.path is None:
                    #print('Path is cooked.')
                    return 
                break
            self.agent = tuple((r, c))   
            self.final_path.append((r, c))
            #print(f"Path {path}")
            #print(f"Agent is at yooyoyo {self.agent}")
                         
    def execution_backwards(self, path):
        #print(f'Print Execution \n {self.h_matrix}')
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        #print(f"Iterative Path {path}")
        for (r, c) in self.path:
            #print(self.agent)
            #print((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if nr in range(self.rows) and nc in range(self.cols) and self.board[nr][nc] == 1:
                    self.planning_board[nr][nc] = 1

            if tuple((r,c)) == self.target:
                print("Path has been found!")
                #print(self.planning_board)
                self.found = True
                return self.final_path
                
            
            if self.board[r][c] == 1:
                #print(f'Blocked Cell at RC: {[r, c]}')
                self.planning_board[r][c] = 1
                #print(f"Parent Dict Before: {self.parent_dict}")
                #print(f"Closed List: {self.closedList}")
                #print(f'Final Path List{self.final_path}')
                #current_cell = (self.agent[0], self.agent[1])
                #while current_cell != tuple(self.target):
                    #print("Trying to find a better path.")
                    #print(f'Cell to be popped {current_cell}')
                    #cell_to_be_popped = current_cell
                    #current_cell = self.parent_dict[current_cell]
                    #self.parent_dict.pop(cell_to_be_popped)
                    #curr_row, curr_col = cell_to_be_popped
                    #print(cell_to_be_popped)
                    #print(self.final_path)
                    #self.final_path.pop(cell_to_be_popped)
                    #self.closedList.pop(cell_to_be_popped)

                #print(f"Parent Dict After: {self.parent_dict}")
                #print(self.agent)
                self.parent_dict = {}
                #print(f'Planning Grid:\n {self.planning_board}')
                #print(f'Execution Grid:\n {self.board}')
                #print(f"Path Before {self.path}")
                
                self.path = self.backwards()
                #print(f"Path totototo {self.path}")
                if self.path is None:
                    print('Path is cooked.')
                    return 
                break
            self.agent = tuple((r, c))   
            self.final_path.append((r, c))
            #print(f"Path {path}")
            #print(f"Agent is at yooyoyo {self.agent}")
        


    def run_visualization(self):
        
        with open('results.txt', 'a') as f:
            
            for i in range(1, 51):
                grid = self.createBoard()
                f.write(f"Board {i} Values:\n")
                self.ForwardAStar_WithBiggerG()
                f.write("Repeated Forward Astar - Larger G-Values:\n")
                f.write("- Expanded Cells: {}\n".format(self.expanded_nodes))
                f.write("- Length of Path: {}\n".format(len(self.final_path)))
                self.reset_board()
                self.ForwardAStar_WithSmallerG()
                f.write("Repeated Forward Astar - Smaller G-Values:\n")
                f.write("- Expanded Cells: {}\n".format(self.expanded_nodes))
                f.write("- Length of Path: {}\n".format(len(self.final_path)))
                self.reset_board()
                self.AdaptiveAStar_WithBiggerG()
                f.write("Adaptive Forward Astar - Larger G-Values:\n")
                f.write("- Expanded Cells: {}\n".format(self.expanded_nodes))
                f.write("- Length of Path: {}\n".format(len(self.final_path)))
                self.reset_board()
                #self.back()
                #f.write("Repeated Backwards Astar:\n")
                #f.write("- Expanded Cells: {}\n".format(self.expanded_nodes))
                #f.write("- Length of Path: {}\n".format(self.len(self.final_path)))
                #f.write("\n")
        
        '''
        #self.ForwardAStar_WithBiggerG()
        #print(f'Forward with Bigger G: {self.expanded_nodes}')
        #print(f'Length of Path: {len(self.final_path)}')

        #self.reset_board()
        #self.ForwardAStar_WithSmallerG()
        #print(f'Forward with Smaller G: {self.expanded_nodes}')
        #print(f'Length of Path: {len(self.final_path)}')

        #self.reset_board()
        #board.createBoard()

        #self.ForwardAStar_WithBiggerG()
        #self.ForwardAStar_WithSmallerG()
        #self.AdaptiveAStar_WithBiggerG()
        self.backwards()
        print(f'Expanded Cells: {self.expanded_nodes}')
        print(f'Length of Path: {len(self.final_path)}')
        
        if self.found is False:
            self.show_popup("No path exists!")
        else:
            # Clear the screen
            self.screen.fill((255, 255, 255))

            # Draw the initial grid
            self.draw_grid(self.final_path)

            # Update the display
            pygame.display.flip()

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Clear the screen
                        self.screen.fill((255, 255, 255))

                        # Draw the final path
                        self.draw_grid(self.final_path, wait_for_click=True)

                        # Update the display
                        pygame.display.flip()

                # Limit frame rate
                self.clock.tick(30)

            pygame.quit()
        
           '''
            

board = Board(101, 101)
grid = board.createBoard()
board.run_visualization()









'''
- Backwards Astar
- Adaptive A*
- Implement on 50 diff grids
'''
