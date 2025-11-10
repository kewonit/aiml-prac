"""
basically just finds the path from point a to point b by guessing which way looks shortest.
it's not always right but it's fast lol
"""
import heapq

# just a bunch of locations connected to each other with travel costs
location_graph = {
    "Dorm": [("Library", 3), ("Cafe", 6)],
    "Library": [("Gym", 4), ("Cafe", 2)],
    "Cafe": [("Gym", 3), ("Lab", 5)],
    "Gym": [("Office", 5)],
    "Lab": [("Office", 2)],
    "Office": []
}

# rough distance guesses from each spot to the end goal
heuristic_distance = {
    "Dorm": 10,
    "Library": 7,
    "Cafe": 6,
    "Gym": 4,
    "Lab": 1,
    "Office": 0
}


def greedy_best_first_search(start_location, goal_location):
    """
    goes through the graph and tries to find a path by always picking
    the location that looks closest to the goal. pretty fast but not always optimal.
    """
    # shove the starting location into the queue
    priority_queue = [(heuristic_distance[start_location], start_location, [start_location], 0)]
    visited_locations = set()
    
    while priority_queue:
        # grab the location that looks closest
        estimated_distance, current_location, current_path, total_cost = heapq.heappop(priority_queue)
        
        # found it, we're done
        if current_location == goal_location:
            return current_path, total_cost
        
        # already been there, skip
        if current_location in visited_locations:
            continue
        visited_locations.add(current_location)
        
        # check out all the neighbors
        for neighbor_location, travel_cost in location_graph.get(current_location, []):
            heapq.heappush(
                priority_queue,
                (
                    heuristic_distance[neighbor_location],
                    neighbor_location,
                    current_path + [neighbor_location],
                    total_cost + travel_cost
                )
            )
    
    # couldn't find anything
    return [], 0


if __name__ == "__main__":
    # find the path from dorm to office
    optimal_path, total_travel_cost = greedy_best_first_search("Dorm", "Office")
    print("Path:", " -> ".join(optimal_path))
    print("Travel cost:", total_travel_cost)
