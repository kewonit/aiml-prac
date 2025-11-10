"""
finds the quickest route through a maze by being smart about which way to go
uses a* to balance actual cost against how far away the goal still is
manhattan distance tells it which direction sucks less
"""
import heapq

# maze grid where 0 is walkable and 1 is a wall you can't go through
game_maze = [
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
]

# starting position and where we need to get to
start_position = (0, 0)
goal_position = (4, 4)


def manhattan_heuristic(current_cell, target_cell):
    # calculates how far away the target is ignoring walls and stuff
    return abs(current_cell[0] - target_cell[0]) + abs(current_cell[1] - target_cell[1])


def astar_maze_search(start_pos, goal_pos):
    # priority queue holds: (f_score, g_cost, current_cell, parent_cell)
    # f_score is the total estimate, g_cost is what we've paid so far
    open_frontier = [(manhattan_heuristic(start_pos, goal_pos), 0, start_pos, None)]
    closed_cells = {}
    lowest_cost_found = {start_pos: 0}
    
    while open_frontier:
        # grab the cell that looks most promising
        f_score, g_cost, current_cell, parent_cell = heapq.heappop(open_frontier)
        
        # skip if we already processed this spot
        if current_cell in closed_cells:
            continue
        closed_cells[current_cell] = parent_cell
        
        # made it to the goal so we're done searching
        if current_cell == goal_pos:
            break
        
        # check all four directions we can move: up down left right
        for row_change, col_change in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_row = current_cell[0] + row_change
            next_col = current_cell[1] + col_change
            next_cell = (next_row, next_col)
            
            # make sure we're not hitting walls or going out of bounds
            if (0 <= next_row < len(game_maze) and 
                0 <= next_col < len(game_maze[0]) and 
                game_maze[next_row][next_col] == 0):
                
                # cost to reach this new cell is one step more than where we came from
                new_cost = g_cost + 1
                
                # only worth checking if it's cheaper than any path we found before
                if new_cost < lowest_cost_found.get(next_cell, float("inf")):
                    lowest_cost_found[next_cell] = new_cost
                    # combine actual cost with heuristic guess to rank it
                    f_score_estimate = new_cost + manhattan_heuristic(next_cell, goal_pos)
                    heapq.heappush(open_frontier, (f_score_estimate, new_cost, next_cell, current_cell))
    
    # rebuild the path from start to goal using parent pointers
    if goal_pos not in closed_cells:
        return [], 0
    
    final_path = []
    current = goal_pos
    while current:
        final_path.append(current)
        current = closed_cells[current]
    
    final_path.reverse()
    return final_path, len(final_path) - 1


if __name__ == "__main__":
    # find the most cost-efficient path from start to goal
    optimal_route, total_movement_cost = astar_maze_search(start_position, goal_position)
    print("Path found:", optimal_route)
    print("Total movement cost:", total_movement_cost)
