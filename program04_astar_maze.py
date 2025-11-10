"""A* on another tiny maze because huge ones scare me.
Manhattan distance guides the search while costs stay uniform.
We keep parents so we can rebuild the winning path.
"""
import heapq

grid = [
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar():
    open_set = [(manhattan(start, goal), 0, start, None)]
    came_from = {}
    best_cost = {start: 0}
    while open_set:
        f, g, node, parent = heapq.heappop(open_set)
        if node in came_from:
            continue
        came_from[node] = parent
        if node == goal:
            break
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                step_cost = g + 1
                if step_cost < best_cost.get((nx, ny), float("inf")):
                    best_cost[(nx, ny)] = step_cost
                    score = step_cost + manhattan((nx, ny), goal)
                    heapq.heappush(open_set, (score, step_cost, (nx, ny), node))
    if goal not in came_from:
        return []
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = came_from[cur]
    return list(reversed(path))


if __name__ == "__main__":
    route = astar()
    print("A* path:", route)
    print("Total steps:", len(route) - 1 if route else 0)
