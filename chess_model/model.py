
from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F


import brevitas.nn as qnn

from torch.optim import Adam 

import pytorch_lightning as pl

from chess_model.utils import piece_available, row_letters, column_letters

from torchmetrics.classification import MulticlassAccuracy

class ConvModel(pl.LightningModule):
    """Deep Convolutional Model.

    Note: In FHE, MaxPooling operations are quite slow in FHE.
    It's better to replace them by using AveragePooling instead,
    if it doesn't drop the accuracy of the model.
    """    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 36, kernel_size=(3,3), stride=1, padding=1)
        self.act1  = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))


        # nn.BatchNorm2d(30)
        # nn.SELU()

        self.conv2 = nn.Conv2d(36, 36, kernel_size=(3,3), stride=1, padding=1)
        self.act2  = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2))
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
         
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(144, 64)
        self.act3 = nn.ReLU()

        self.out = nn.Linear(64, 6+8+8)

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
        # piece  = F.relu(self.fc_piece(x))
        # row    = F.relu(self.fc_row(x))
        # column = F.relu(self.fc_column(x))
        # return piece, row, column
        return self.out(x)

    def configure_optimizers(self):
        return Adam(self.parameters())
    
    def training_step(self, train_batch, batch_idx):
        x, (y_piece, y_row, y_col) = train_batch
        # x = x.view(x.size(0), -1)
        

        y_pred = self(x)
        pred_piece, pred_row, pred_col = torch.split(y_pred, [6, 8, 8], dim=1)
        # pred_piece, pred_row, pred_col = self(x)

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
        

        y_pred = self(x)
        pred_piece, pred_row, pred_col = torch.split(y_pred, [6, 8, 8], dim=1)
        # pred_piece, pred_row, pred_col = self(x)

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
    
class QATConvolutionNetwork(pl.LightningModule):

    def __init__(self, n_bits: int = 4) -> None:
        
        a_bits = n_bits
        w_bits = n_bits

        self.q1 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(12, 36, (3, 3), stride=1, padding=1, weight_bit_width=w_bits)
        self.relu1 = qnn.QuantReLU()
        self.pool1 = qnn.QuantAvgPool2d(kernel_size=(2,2))

        self.q2 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(36, 36, (3,3), stride=1, padding=1, weight_bit_width=w_bits)
        self.relu2 = qnn.QuantReLU()
        self.pool2 = qnn.QuantAvgPool2d(kernel_size=(2,2))

        self.q3 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(144, 64, weight_quant=w_bits)
        self.relu3 = qnn.QuantReLU()
        
        self.q3 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(64, 6+8+8, weight_quant=w_bits)

        