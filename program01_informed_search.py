"""This tiny script pretends to be a delivery kid picking fastest path by peeking at a cheap heuristic.
The graph is a couple of city spots, heuristic guesses distance to the office.
Greedy best first search pops the place that feels closest to the goal.
"""
import heapq

graph = {
    "Dorm": [("Library", 3), ("Cafe", 6)],
    "Library": [("Gym", 4), ("Cafe", 2)],
    "Cafe": [("Gym", 3), ("Lab", 5)],
    "Gym": [("Office", 5)],
    "Lab": [("Office", 2)],
    "Office": []
}

heuristic = {
    "Dorm": 10,
    "Library": 7,
    "Cafe": 6,
    "Gym": 4,
    "Lab": 1,
    "Office": 0
}


def greedy_best_first(start, goal):
    queue = [(heuristic[start], start, [start], 0)]
    seen = set()
    while queue:
        guess, place, path, spent = heapq.heappop(queue)
        if place == goal:
            return path, spent
        if place in seen:
            continue
        seen.add(place)
        for nxt, cost in graph.get(place, []):
            heapq.heappush(queue, (heuristic[nxt], nxt, path + [nxt], spent + cost))
    return [], 0


if __name__ == "__main__":
    final_path, final_cost = greedy_best_first("Dorm", "Office")
    print("Path:", " -> ".join(final_path))
    print("Travel cost:", final_cost)
