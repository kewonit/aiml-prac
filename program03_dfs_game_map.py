"""
depth-first search traversal of a game world map.
explores every room starting from entrance, marking visited rooms so we don't loop,
and shows the order we discovered each room.
"""

# the game map - each room connects to other rooms
game_world_graph = {
    "Entrance": ["Hall", "Kitchen"],
    "Hall": ["Armory", "Garden"],
    "Kitchen": ["Pantry"],
    "Pantry": [],
    "Armory": ["Boss"],
    "Garden": ["Boss"],
    "Boss": []
}

# where we're trying to reach
target_room = "Boss"


def dfs_explore_game_map(start_room, target_destination):
    """
    uses depth-first search to explore all rooms and find paths to the target.
    keeps track of visited rooms and records the order we discovered them.
    """
    # stack holds tuples of (current_room, path_taken_so_far)
    exploration_stack = [(start_room, [start_room])]
    traversal_order = []
    visited_rooms = set()
    paths_to_target = []
    
    while exploration_stack:
        # pop the last room we were exploring
        current_room, current_path = exploration_stack.pop()
        
        # skip if we already visited this room
        if current_room in visited_rooms:
            continue
        
        visited_rooms.add(current_room)
        traversal_order.append((current_room, list(current_path)))
        
        # if we found the target, save this path
        if current_room == target_destination:
            paths_to_target.append(current_path)
        
        # explore neighbors (reversed to maintain left-to-right order with stack)
        for adjacent_room in reversed(game_world_graph.get(current_room, [])):
            exploration_stack.append((adjacent_room, current_path + [adjacent_room]))
    
    return traversal_order, paths_to_target


if __name__ == "__main__":
    # explore the game map from entrance to boss
    traversal_sequence, target_paths = dfs_explore_game_map("Entrance", target_room)
    
    # show traversal order
    print("=== DFS Traversal Order ===")
    for discovery_number, (room_name, path_taken) in enumerate(traversal_sequence, 1):
        print(f"{discovery_number}. Reached {room_name} via {' -> '.join(path_taken)}")
    
    # show all paths to target
    print(f"\n=== Paths to Target ({target_room}) ===")
    if target_paths:
        for path_number, target_path in enumerate(target_paths, 1):
            print(f"Path {path_number}: {' -> '.join(target_path)}")
    else:
        print("No path found to target!")
