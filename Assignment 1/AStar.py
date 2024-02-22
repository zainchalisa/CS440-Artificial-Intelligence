import numpy
import random
import heapq
import pygame
import time

class Board:
    def __init__(self, rows, cols):
        self.parent_dict = {}
        self.final_path = []
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
    
    def draw_grid(self, path):
        
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
    
    def ForwardAStar_WithBiggerG(self) -> list:
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        self.openList = []
        heapq.heapify(self.openList)
        found_destination = False


        # First execution of planning
        if self.agent[0] == self.initial[0] and self.agent[1] == self.initial[1]: 
            heapq.heappush(self.openList, (self.h_matrix[self.agent[0]][self.agent[1]], 0, self.calculateHeuristic(self.initial[0], self.initial[1]) * -1, self.initial[0], self.initial[1], -1, -1))
        
        # Agent is somewhere else besides initial cell
        else:
            h_val = self.h_matrix[self.agent[0]][self.agent[1]]
            g_val = self.closedList[(self.agent[0], self.agent[1])]
            f_val = h_val + g_val 
            parentRow, parentCol = self.parent_dict[(self.agent[0], self.agent[1])]
            heapq.heappush(self.openList, (f_val, g_val, h_val, self.agent[0], self.agent[1], parentRow, parentCol))

        while self.openList and not found_destination:
            #print(self.openList)
            fValue, gValue, _, childRow, childCol, parentRow, parentCol = heapq.heappop(self.openList)
            self.closedList[(childRow, childCol)] = gValue
            if (childRow, childCol) not in self.parent_dict.keys():
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
            path = [(self.target[0], self.target[1])]
            current_cell = (self.target[0], self.target[1])

            while current_cell != tuple(self.agent):
                print(f'Current Cell: {current_cell}')
                current_cell = self.parent_dict[current_cell]
                path.append(current_cell)

            path.reverse()
            print(path)
            self.execution(path)
            
        else:
            print('no path found')
            return None

    def execution(self, path):
        
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        for (r, c) in path:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if nr in range(self.rows) and nc in range(self.cols) and self.board[nr][nc] == 1:
                    self.planning_board[nr][nc] == 1

            if self.board[r][c] == 1:
                self.planning_board[r][c] == 1
                print(f"Parent Dict Before: {self.parent_dict}")
                current_cell = (self.target[0], self.target[1])
                while current_cell != tuple(self.agent):
                    print("Trying to find a better path.")
                    cell_to_be_popped = current_cell
                    current_cell = self.parent_dict[current_cell]
                    self.parent_dict.pop(cell_to_be_popped)
                    self.closedList.pop(cell_to_be_popped)
                    
                print(f"Parent Dict Before: {self.parent_dict}")
                print(self.agent)
                path = self.ForwardAStar_WithBiggerG()
                if path is None:
                    print('Path is cooked.')
                    return 
                break

            self.agent = list((r, c))
            self.final_path.append((r, c))
            if (r, c) == self.target:
                return "Path has been found!"
                
            
            
    

            
            


    def run_visualization(self):
        #path = board.ForwardAStar_WithSmallerG(grid)
        
        #print(path)
       
        board.ForwardAStar_WithBiggerG()


        # forward_time = board.forward_v_backwards(grid, self.initial, self.target)
        # print(f'Forward Search: {forward_time}')
        #backward_time = board.forward_v_backwards(grid, self.target, self.initial)
        #print(f'Backwards Search: {backward_time}')
       
        if self.final_path is None:
            self.show_popup("No path found!")
        else:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # Clear the screen
                self.screen.fill((255, 255, 255))

                # Draw the grid
                self.draw_grid(self.final_path)

                # Update the display
                pygame.display.flip()

                # Limit frame rate
                self.clock.tick(30)

            pygame.quit()

        

board = Board(10, 10)
board.createBoard()
board.run_visualization()









'''
- Backwards Astar
- Adaptive A*
- Implement on 50 diff grids
'''