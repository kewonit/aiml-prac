"""Classic two jug puzzle with BFS.
Jugs sizes are 4 and 3, goal is 2 in the big jug because teacher asked that in lab.
States are (big, small) and we try pouring all the obvious actions.
"""
from collections import deque

CAP_A = 4
CAP_B = 3
GOAL = 2


def neighbors(a, b):
    options = []
    options.append((CAP_A, b))
    options.append((a, CAP_B))
    options.append((0, b))
    options.append((a, 0))
    pour = min(a, CAP_B - b)
    options.append((a - pour, b + pour))
    pour = min(b, CAP_A - a)
    options.append((a + pour, b - pour))
    return options


def solve():
    q = deque([((0, 0), [])])
    seen = {(0, 0)}
    while q:
        (a, b), path = q.popleft()
        if a == GOAL:
            return path + [(a, b)]
        for nxt in neighbors(a, b):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [(a, b)]))
    return []


if __name__ == "__main__":
    solution = solve()
    for state in solution:
        print(state)
