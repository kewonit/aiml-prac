"""Solving 8-puzzle with plain BFS because it is easy to write.
State is a tuple of 9 numbers, 0 is the blank. We explore neighbors until the goal pops.
I only run it on a near-solved board so it finishes quickly.
"""
from collections import deque

goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
start = (1, 2, 3, 4, 5, 6, 0, 7, 8)

moves = {
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


def swap(state, i, j):
    lst = list(state)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


def bfs_solve():
    q = deque([(start, [])])
    seen = {start}
    while q:
        state, path = q.popleft()
        if state == goal:
            return path + [state]
        zero = state.index(0)
        for nxt in moves[zero]:
            new_state = swap(state, zero, nxt)
            if new_state not in seen:
                seen.add(new_state)
                q.append((new_state, path + [state]))
    return []


if __name__ == "__main__":
    result = bfs_solve()
    print("States visited to finish:")
    for s in result:
        print(s)
    print("Moves needed:", len(result) - 1 if result else 0)
