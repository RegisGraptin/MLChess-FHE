import pandas as pd
import torch
import torchvision

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

from chess_model.model import ConvModel, SimpleModel
from chess_model.dataset import ChessDataset


def data_train_test_split(filename: str) -> tuple[str, str]:
    train_filename = "/tmp/train.csv"
    test_filename = "/tmp/test.csv"

    df = pd.read_csv(filename)
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(train_filename)
    test.to_csv(test_filename)

    return train_filename, test_filename

if __name__ == "__main__":

    train, test = data_train_test_split("./data/samples_data.csv")

    training_chess = ChessDataset(train)
    testing_chess  = ChessDataset(test)
    
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    batch_size   = 32
    train_loader = torch.utils.data.DataLoader(training_chess, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader   = torch.utils.data.DataLoader(testing_chess, batch_size=batch_size, num_workers=6)

    model = SimpleModel()

    # # training
    # trainer = pl.Trainer(max_epochs=5)
    # trainer.fit(model, train_loader, val_loader)


    train_loader
