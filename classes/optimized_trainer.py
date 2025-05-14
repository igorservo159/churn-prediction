
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from architecture import Architecture

class OptimizedTrainer:
    def __init__(self, data_path='clean_data.pkl', batch_size=64, lr=0.001, test_size=0.2, seed=13):
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.test_size = test_size
        self.seed = seed

        self.scaler = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.arch = None

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            df = pickle.load(f)
        X = df.drop(columns=["Churn"]).values
        y = df["Churn"].values
        return X, y

    def process_data(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        return X_train, X_val, y_train, y_val

    def build_dataloaders(self, X_train, y_train, X_val, y_val):
        x_train_tensor = torch.as_tensor(X_train).float()
        y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()
        x_val_tensor = torch.as_tensor(X_val).float()
        y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2)

    def build_model(self, input_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self, n_epochs=50):
        torch.manual_seed(self.seed)

        X, y = self.load_data()
        X_train, X_val, y_train, y_val = self.process_data(X, y)
        self.build_dataloaders(X_train, y_train, X_val, y_val)
        self.build_model(input_dim=X.shape[1])

        self.arch = Architecture(self.model, self.loss_fn, self.optimizer)
        self.arch.set_loaders(self.train_loader, self.val_loader)
        self.arch.set_seed(42)
        self.arch.train(n_epochs=n_epochs)

        return self.arch, self.scaler
