"""A tiny tic tac toe simulation.
I hard-code a list of moves for X and O so we see a complete playthrough without typing.
Just shows win detection and board printing.
"""

board = [" "] * 9

win_lines = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


def print_board():
    print(f"{board[0]}|{board[1]}|{board[2]}")
    print("-+-+-")
    print(f"{board[3]}|{board[4]}|{board[5]}")
    print("-+-+-")
    print(f"{board[6]}|{board[7]}|{board[8]}")


def winner():
    for a, b, c in win_lines:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    if " " not in board:
        return "Tie"
    return None


if __name__ == "__main__":
    scripted_moves = [0, 3, 1, 4, 2]
    symbols = ["X", "O"]
    turn = 0
    for move in scripted_moves:
        board[move] = symbols[turn]
        print_board()
        champ = winner()
        if champ:
            print("Result:", champ)
            break
        print()
        turn = 1 - turn
