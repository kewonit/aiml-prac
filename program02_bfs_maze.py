"""Basic BFS on a toy maze grid.
0 means walkable, 1 means wall. We keep it tiny so my brain doesn’t melt.
The queue pops cells level by level until the goal shows up.
"""
from collections import deque

grid = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def bfs_path():
    q = deque([(start, [start])])
    seen = {start}
    while q:
        spot, path = q.popleft()
        if spot == goal:
            return path
        for dx, dy in dirs:
            nx, ny = spot[0] + dx, spot[1] + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                if grid[nx][ny] == 0 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
    return []


if __name__ == "__main__":
    path = bfs_path()
    print("BFS steps:", path)
    print("Steps taken:", len(path) - 1 if path else 0)
