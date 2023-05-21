
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.optim import Adam 

import pytorch_lightning as pl

from chess_model.utils import piece_available, row_letters, column_letters

from torchmetrics.classification import MulticlassAccuracy

class ConvModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 36, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(36, 36, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(144, 64)
        self.act3 = nn.ReLU()
 
        self.fc_piece  = nn.Linear(64, 6)
        self.fc_row    = nn.Linear(64, 8)
        self.fc_column = nn.Linear(64, 8)

        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion3 = nn.CrossEntropyLoss()


        self.metric_piece = MulticlassAccuracy(num_classes=6)
        self.metric_row = MulticlassAccuracy(num_classes=8)
        self.metric_col = MulticlassAccuracy(num_classes=8)


 
    def forward(self, x):
        
        x = self.act1(self.conv1(x))
        x = self.pool1(x)

        x = self.act2(self.conv2(x))
        x = self.pool2(x)

        x = self.flat(x)
        x = self.act3(self.fc3(x))
        
        # input 512, output 10
        piece  = F.relu(self.fc_piece(x))
        row    = F.relu(self.fc_row(x))
        column = F.relu(self.fc_column(x))

        return piece, row, column

    def configure_optimizers(self):
        return Adam(self.parameters())
    
    def training_step(self, train_batch, batch_idx):
        x, (y_piece, y_row, y_col) = train_batch
        # x = x.view(x.size(0), -1)
        
        pred_piece, pred_row, pred_col = self(x)

        loss_p = self.criterion1(pred_piece, y_piece)
        loss_r = self.criterion2(pred_row, y_row)
        loss_c = self.criterion3(pred_col, y_col)

        total_loss = 0.3 * loss_p + 0.3 * loss_r + 0.3 * loss_c

        self.log('train_loss', total_loss, prog_bar=True)

        accuracy_piece = self.metric_piece(torch.argmax(pred_piece, dim=1), torch.argmax(y_piece, dim=1))
        accuracy_row   = self.metric_row(torch.argmax(pred_row, dim=1), torch.argmax(y_row, dim=1))
        accuracy_col   = self.metric_col(torch.argmax(pred_col, dim=1), torch.argmax(y_col, dim=1))

        self.log('train_accuracy_piece', accuracy_piece, prog_bar=True)
        self.log('train_accuracy_row', accuracy_row, prog_bar=True)
        self.log('train_accuracy_col', accuracy_col, prog_bar=True)

        return total_loss
    
    def validation_step(self, val_batch, batch_idx):
        x, (y_piece, y_row, y_col) = val_batch
        # x = x.view(x.size(0), -1)
        
        pred_piece, pred_row, pred_col = self(x)

        loss_p = self.criterion1(pred_piece, y_piece)
        loss_r = self.criterion2(pred_row, y_row)
        loss_c = self.criterion3(pred_col, y_col)

        total_loss = 0.3 * loss_p + 0.3 * loss_r + 0.3 * loss_c

        self.log('val_loss', total_loss, prog_bar=True)

        accuracy_piece = self.metric_piece(torch.argmax(pred_piece, dim=1), torch.argmax(y_piece, dim=1))
        accuracy_row   = self.metric_row(torch.argmax(pred_row, dim=1), torch.argmax(y_row, dim=1))
        accuracy_col   = self.metric_col(torch.argmax(pred_col, dim=1), torch.argmax(y_col, dim=1))

        self.log('val_accuracy_piece', accuracy_piece, prog_bar=True)
        self.log('val_accuracy_row', accuracy_row, prog_bar=True)
        self.log('val_accuracy_col', accuracy_col, prog_bar=True)

        return total_loss
    
    
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
 
        self.fc1 = nn.Linear(768, 512)
        self.act1 = nn.ReLU()

        self.fc2   = nn.Linear(512, 256)
        self.act2  = nn.ReLU()

        self.fc3   = nn.Linear(256, 128)
        self.act3  = nn.ReLU()
 
        self.fc_piece  = nn.Linear(128, 6)
        self.fc_row    = nn.Linear(128, 8)
        self.fc_column = nn.Linear(128, 8)

        # self.ac_p = nn.Softmax(dim=1)
        # self.ac_r = nn.Softmax(dim=1)
        # self.ac_c = nn.Softmax(dim=1)

        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion3 = nn.CrossEntropyLoss()


 
    def forward(self, x):
        x = self.flat(x)
        x = self.act1(self.fc1(x))
        
        x = self.act2(self.fc2(x))

        x = self.act3(self.fc3(x))
        
        # input 512, output 10
        # piece  = self.ac_p(self.fc_piece(x))
        # row    = self.ac_r(self.fc_row(x))
        # column = self.ac_c(self.fc_column(x))


        piece  = self.fc_piece(x)
        row    = self.fc_row(x)
        column = self.fc_column(x)

        return piece, row, column

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, (y_piece, y_row, y_col) = train_batch
        # x = x.view(x.size(0), -1)
        
        pred_piece, pred_row, pred_col = self(x)

        loss_p = self.criterion1(pred_piece, y_piece)
        loss_r = self.criterion2(pred_row, y_row)
        loss_c = self.criterion3(pred_col, y_col)

        total_loss = loss_p + loss_r + loss_c

        self.log('train_loss', total_loss, prog_bar=True)

        accuracy_piece = torch.sum(torch.argmax(pred_piece, dim=1) == torch.argmax(y_piece, dim=1)) / len(y_piece)
        accuracy_row   = torch.sum(torch.argmax(pred_row, dim=1) == torch.argmax(y_row, dim=1)) / len(y_piece)
        accuracy_col   = torch.sum(torch.argmax(pred_col, dim=1) == torch.argmax(y_col, dim=1)) / len(y_piece)

        self.log('train_accuracy_piece', accuracy_piece, prog_bar=True)
        self.log('train_accuracy_row', accuracy_row, prog_bar=True)
        self.log('train_accuracy_col', accuracy_col, prog_bar=True)

        return total_loss
    
    def validation_step(self, val_batch, batch_idx):
        x, (y_piece, y_row, y_col) = val_batch
        # x = x.view(x.size(0), -1)
        
        pred_piece, pred_row, pred_col = self(x)

        loss_p = F.cross_entropy(pred_piece, y_piece)
        loss_r = F.cross_entropy(pred_row, y_row)
        loss_c = F.cross_entropy(pred_col, y_col)

        total_loss = loss_p + loss_r + loss_c

        self.log('val_loss', total_loss, prog_bar=True)

        accuracy_piece = torch.sum(torch.argmax(pred_piece, dim=1) == torch.argmax(y_piece, dim=1)) / len(y_piece)
        accuracy_row   = torch.sum(torch.argmax(pred_row, dim=1) == torch.argmax(y_row, dim=1)) / len(y_piece)
        accuracy_col   = torch.sum(torch.argmax(pred_col, dim=1) == torch.argmax(y_col, dim=1)) / len(y_piece)

        self.log('val_accuracy_piece', accuracy_piece, prog_bar=True)
        self.log('val_accuracy_row', accuracy_row, prog_bar=True)
        self.log('val_accuracy_col', accuracy_col, prog_bar=True)

        return total_loss
    