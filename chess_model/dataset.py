
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from chess_model.utils import board_encoding, chess_columns, chess_rows, chess_pieces

class ChessDataset(Dataset):
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)

        self.df["target_column"] = self.df["to_square"].apply(lambda x: x[0])
        self.df["target_row"]    = self.df["to_square"].apply(lambda x: x[1])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        X = board_encoding(data["board"])

        piece, row, column = (
            np.array(chess_pieces[data["piece"]]),
            np.array(chess_rows[data["target_row"]]),
            np.array(chess_columns[data["target_column"]])
        )

        return torch.tensor(X).type(torch.FloatTensor), (
            torch.tensor(piece).type(torch.FloatTensor), 
            torch.tensor(row).type(torch.FloatTensor), 
            torch.tensor(column).type(torch.FloatTensor)
        )
