"""Tower of Hanoi with lazy recursion and three plates.
I just print each move, not going fancy with graphics.
"""

moves = []


def solve(n, src, dst, aux):
    if n == 0:
        return
    solve(n - 1, src, aux, dst)
    moves.append((src, dst))
    solve(n - 1, aux, dst, src)


if __name__ == "__main__":
    solve(3, "A", "C", "B")
    for step, (a, b) in enumerate(moves, 1):
        print(f"Move {step}: {a} -> {b}")
    print("Total moves:", len(moves))
