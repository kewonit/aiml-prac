"""
tower of hanoi game - basically moving disks from one peg to another
without breaking the rules (can't put a bigger disk on a smaller one)
"""

# list to keep track of all the moves we make
all_moves_recorded = []


def solve_hanoi_tower(number_of_disks, source_peg, destination_peg, auxiliary_peg):
    """
    recursively solves tower of hanoi by moving disks one at a time
    the trick is using the middle peg as a temp spot while moving stuff around
    """
    # base case: if no disks to move, we're done
    if number_of_disks == 0:
        return
    
    # move n-1 disks from source to auxiliary using destination as temp
    solve_hanoi_tower(number_of_disks - 1, source_peg, auxiliary_peg, destination_peg)
    
    # move the biggest disk from source to destination
    all_moves_recorded.append((source_peg, destination_peg))
    
    # move n-1 disks from auxiliary to destination using source as temp
    solve_hanoi_tower(number_of_disks - 1, auxiliary_peg, destination_peg, source_peg)


if __name__ == "__main__":
    # number of disks we wanna move around
    number_of_disks = 3
    source_peg = "A"
    destination_peg = "C"
    auxiliary_peg = "B"
    
    # solve it
    solve_hanoi_tower(number_of_disks, source_peg, destination_peg, auxiliary_peg)
    
    # print out each move nicely
    for move_number, (from_peg, to_peg) in enumerate(all_moves_recorded, 1):
        print(f"Move {move_number}: {from_peg} -> {to_peg}")
    
    # show how many moves it took (should be 2^n - 1)
    print(f"Total moves: {len(all_moves_recorded)}")
