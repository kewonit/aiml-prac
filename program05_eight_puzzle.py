"""
the 8-puzzle game. got 8 numbered tiles and 1 blank space arranged in a 3x3 grid.
we're trying to rearrange them from a scrambled state into the goal state.
uses bfs to find the sequence of moves needed. state is a tuple of 9 numbers where 0 = blank.
"""
from collections import deque

# the target arrangement we're trying to reach
goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# starting position of the puzzle (already close to solved so it's fast)
starting_state = (1, 2, 3, 4, 5, 6, 0, 7, 8)

# neighbor mapping for a 3x3 grid. each position can swap with certain adjacent positions
# layout is: [0][1][2]
#           [3][4][5]
#           [6][7][8]
adjacent_moves_per_position = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7]
}


def swap_positions(puzzle_state, blank_position, adjacent_position):
    """swap the blank with an adjacent tile to create a new state"""
    state_list = list(puzzle_state)
    state_list[blank_position], state_list[adjacent_position] = state_list[adjacent_position], state_list[blank_position]
    return tuple(state_list)


def solve_puzzle_with_bfs():
    """
    uses breadth-first search (BFS) to find the shortest path from start to goal.
    keeps track of visited states so we don't go in circles.
    """
    # queue stores tuples of (current_state, path_taken_so_far)
    search_queue = deque([(starting_state, [])])
    visited_states = {starting_state}
    
    while search_queue:
        current_puzzle_state, moves_path = search_queue.popleft()
        
        # if we hit the goal, return all states we went through
        if current_puzzle_state == goal_state:
            return moves_path + [current_puzzle_state]
        
        # find where the blank (0) is in the current state
        blank_position = current_puzzle_state.index(0)
        
        # try swapping with each adjacent position
        for adjacent_position in adjacent_moves_per_position[blank_position]:
            new_puzzle_state = swap_positions(current_puzzle_state, blank_position, adjacent_position)
            
            # only explore if we haven't seen this state before
            if new_puzzle_state not in visited_states:
                visited_states.add(new_puzzle_state)
                search_queue.append((new_puzzle_state, moves_path + [current_puzzle_state]))
    
    return []


if __name__ == "__main__":
    solution_states = solve_puzzle_with_bfs()
    
    print("all puzzle states from start to finish:")
    for state_index, puzzle_state in enumerate(solution_states):
        print(f"move {state_index}: {puzzle_state}")
    
    print(f"\ntotal moves needed: {len(solution_states) - 1 if solution_states else 0}")
