"""
water jug problem using bfs. you gotta figure out how to get exactly 2 liters in the 4L jug.
the catch is you've got a 4 liter jug and a 3 liter jug, and you can only fill, empty, or pour between them.
it's basically a state space search where each state is how much water is in each jug.
"""
from collections import deque

# jug capacities in liters
BIG_JUG_CAPACITY = 4
SMALL_JUG_CAPACITY = 3

# we're trying to get exactly this much in the big jug
TARGET_AMOUNT = 2


def get_possible_next_states(big_jug_amount, small_jug_amount):
    """
    from a given state (how much in each jug), what moves can we make?
    like fill one jug completely, empty one, or pour between them.
    """
    possible_moves = []
    
    # fill the big jug to the top
    possible_moves.append((BIG_JUG_CAPACITY, small_jug_amount))
    
    # fill the small jug to the top
    possible_moves.append((big_jug_amount, SMALL_JUG_CAPACITY))
    
    # empty the big jug completely
    possible_moves.append((0, small_jug_amount))
    
    # empty the small jug completely
    possible_moves.append((big_jug_amount, 0))
    
    # pour from big jug to small jug (until small is full or big is empty)
    amount_to_pour_big_to_small = min(big_jug_amount, SMALL_JUG_CAPACITY - small_jug_amount)
    possible_moves.append((big_jug_amount - amount_to_pour_big_to_small, small_jug_amount + amount_to_pour_big_to_small))
    
    # pour from small jug to big jug (until big is full or small is empty)
    amount_to_pour_small_to_big = min(small_jug_amount, BIG_JUG_CAPACITY - big_jug_amount)
    possible_moves.append((big_jug_amount + amount_to_pour_small_to_big, small_jug_amount - amount_to_pour_small_to_big))
    
    return possible_moves


def find_water_jug_solution():
    """
    bfs through all possible jug states. start at empty and empty, try every move,
    and keep going until we find a state where the big jug has exactly the target amount.
    """
    # queue stores tuples of (current_state, path_to_get_here)
    search_queue = deque([((0, 0), [])])
    visited_states = {(0, 0)}
    
    while search_queue:
        current_state, path_taken = search_queue.popleft()
        big_jug_amount, small_jug_amount = current_state
        
        # check if we reached the goal
        if big_jug_amount == TARGET_AMOUNT:
            return path_taken + [current_state]
        
        # try all possible moves from this state
        for next_state in get_possible_next_states(big_jug_amount, small_jug_amount):
            if next_state not in visited_states:
                visited_states.add(next_state)
                search_queue.append((next_state, path_taken + [current_state]))
    
    # no solution found
    return []


if __name__ == "__main__":
    solution_path = find_water_jug_solution()
    
    if solution_path:
        print("solution found! here's the sequence of jug states (big, small):")
        for step_number, jug_state in enumerate(solution_path):
            print(f"step {step_number}: {jug_state}")
    else:
        print("couldn't find a solution")
