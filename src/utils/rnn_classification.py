"""
RNN Classification Module
This module provides a shallow Recurrent Neural Network (RNN) based classifier
that supports multiple RNN architectures (LSTM, GRU, RNN) for sequence classification tasks.
Classes:
    ShallowRNN: A scikit-learn compatible estimator that wraps RNN models for classification.
        Features early stopping, configurable architecture, and support for CUDA acceleration.
    LSTMModel: PyTorch LSTM-based neural network model that processes sequential input
        and produces classification outputs using the last hidden state.
    GRUModel: PyTorch GRU-based neural network model that processes sequential input
        and produces classification outputs using the last hidden state.
    RNNModel: PyTorch vanilla RNN-based neural network model that processes sequential input
        and produces classification outputs using the last hidden state.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator

random.seed(0)
torch.manual_seed(0)

class ShallowRNN(BaseEstimator):
    def __init__(self, input_dim, hidden_dim, output_dim, model='lstm', num_layers=1, learning_rate=0.001, patience=5, max_epochs=100, batch_size=32, verbose=False):
        """
        Parameters:
        - input_dim: int, the number of input features
        - hidden_dim: int, the number of hidden units in the LSTM
        - output_dim: int, the number of output features
        - num_layers: int, the number of LSTM layers
        - learning_rate: float, the learning rate for the optimizer
        - num_epochs: int, the number of training epochs
        - batch_size: int, the batch size for training
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        # Initialize the LSTM model
        if model == 'lstm':
            self.model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(self.device)
        elif model == 'gru':
            self.model = GRUModel(input_dim, hidden_dim, output_dim, num_layers).to(self.device)
        elif model == 'rnn':
            self.model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0)

    def fit(self, X, y):
        """Train the LSTM model."""
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                epoch_corr = (torch.argmax(F.softmax(outputs, dim=1), dim=1) == batch_y).float().mean()
                epoch_acc += epoch_corr

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.max_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {epoch_acc/len(dataloader):.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                    
                break
        

    def predict(self, X, return_scores=False):
        """Make predictions using the trained LSTM model."""
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scores_ = F.softmax(self.model(X), dim=1)
            predictions = torch.argmax(scores_, dim=1).cpu().numpy()
            scores = scores_.cpu().numpy()

        if return_scores:
            return predictions, scores
        else:
            return predictions

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out
    
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.lstm = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out
