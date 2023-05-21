import numpy as np

chess_dict = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

# Piece encoder
piece_available = "PBQNKR"
chess_pieces = {piece_available[i]:[int(j == i) for j in range(len(piece_available))] for i in range(len(piece_available))}


# Compute one hot encoding for letter column
column_letters = "abcdefgh"
chess_columns = {column_letters[i]:[int(j == i) for j in range(len(column_letters))] for i in range(len(column_letters))}


# Compute one hot encoding for row
row_letters = "12345678"
chess_rows = {row_letters[i]:[int(j == i) for j in range(len(row_letters))] for i in range(len(row_letters))}

def board_encoder(boards: list[str]) -> np.array:
    return np.array([board_encoding(board) for board in boards])


def board_encoding(board: str) -> np.array:
    encoding = []
    lines = board.splitlines()
    for line in lines:
        chars = line.split(' ')
        for char in chars:
            encoding.append(chess_dict[char])
    
    return np.array(encoding).reshape((-1, 8, 8))
