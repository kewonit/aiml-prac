"""implementation of tic-tac-toe game.
simulates a complete game with predefined moves for both players.
detects wins, ties, and displays the board state.
"""

# the 3x3 board represented as a list of 9 positions (indexes 0-8)
game_board = [" "] * 9

# all possible winning combinations (3 in a row, column, or diagonal)
winning_combinations = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6)               # diagonals
]


def display_game_board():
    """renders the current board state in a nice 3x3 grid format"""
    print(f"{game_board[0]}|{game_board[1]}|{game_board[2]}")
    print("-+-+-")
    print(f"{game_board[3]}|{game_board[4]}|{game_board[5]}")
    print("-+-+-")
    print(f"{game_board[6]}|{game_board[7]}|{game_board[8]}")


def check_game_winner():
    """checks if someone won or if it's a tie. returns the winner symbol, 'tie', or none"""
    # loop through all winning combinations
    for pos1, pos2, pos3 in winning_combinations:
        # if all three positions match and aren't empty, we got a winner
        if game_board[pos1] != " " and game_board[pos1] == game_board[pos2] == game_board[pos3]:
            return game_board[pos1]
    
    # if board is full with no winner, it's a tie
    if " " not in game_board:
        return "Tie"
    
    # game still ongoing
    return None


def play_game(predefined_moves):
    """plays out the game with predefined moves and checks win conditions after each move"""
    player_symbols = ["X", "O"]
    current_player_turn = 0
    
    for move_position in predefined_moves:
        # place the current player's symbol at the given position
        game_board[move_position] = player_symbols[current_player_turn]
        
        # show the board after this move
        display_game_board()
        
        # check if game is over
        game_result = check_game_winner()
        if game_result:
            print(f"game result: {game_result}")
            break
        
        print()  # blank line between moves for readability
        # switch to the other player
        current_player_turn = 1 - current_player_turn


if __name__ == "__main__":
    # sequence of moves (position indexes) for a complete game
    moves_sequence = [0, 3, 1, 4, 2]
    play_game(moves_sequence)
