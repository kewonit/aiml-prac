"""
finds a path through a maze using breadth-first search.
0 is walkable, 1 is a wall. it explores level by level until it finds the exit.
"""
from collections import deque

# the maze grid - 0 is open space, 1 is wall
maze_grid = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

# where we start and where we wanna go
start_position = (0, 0)
goal_position = (4, 4)

# up, down, left, right moves
movement_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def find_maze_path_bfs():
    """
    explores the maze level by level to find the shortest path.
    keeps track of visited cells so we don't go in circles.
    """
    # start with the initial position and the path so far
    queue = deque([(start_position, [start_position])])
    visited_cells = {start_position}
    
    while queue:
        # get the next position and the path taken to get there
        current_position, current_path = queue.popleft()
        
        # if we found the goal, return the path
        if current_position == goal_position:
            return current_path
        
        # try moving in each direction
        for row_change, col_change in movement_directions:
            next_row = current_position[0] + row_change
            next_col = current_position[1] + col_change
            next_position = (next_row, next_col)
            
            # make sure it's in bounds and not a wall and not already visited
            if (0 <= next_row < len(maze_grid) and 
                0 <= next_col < len(maze_grid[0]) and
                maze_grid[next_row][next_col] == 0 and 
                next_position not in visited_cells):
                
                visited_cells.add(next_position)
                queue.append((next_position, current_path + [next_position]))
    
    # no path found
    return []


if __name__ == "__main__":
    # find the shortest path through the maze
    shortest_path = find_maze_path_bfs()
    print("BFS steps:", shortest_path)
    print("Steps taken:", len(shortest_path) - 1 if shortest_path else 0)