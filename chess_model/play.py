
import chess
import torch

import torch.nn.functional as F
import torch.nn as nn

from chess_model.utils import board_encoding, piece_available, column_letters, row_letters

def get_model_move(model, board):
    board = str(board)

    X = board_encoding(board)
    X = torch.tensor(X).type(torch.FloatTensor)
    X = torch.unsqueeze(X, dim=0)

    piece, row, col = model(X)
    
    pred_piece = piece_available[torch.argmax(piece, dim=1)]
    pred_row = row_letters[torch.argmax(piece, dim=1)]
    pred_col = column_letters[torch.argmax(piece, dim=1)]

    return pred_piece, pred_row, pred_col


if __name__ == "__main__":
    model = torch.load("./model/model")
    model.eval()

    board = chess.Board()

    piece, row, col = get_model_move(model, board)

    print(piece, row, col)

    move = col + row
    board.push_san(move)

    print(board)


    
    
    