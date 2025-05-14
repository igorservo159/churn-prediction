
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from architecture import Architecture

class Trainer:
    def __init__(self, data_path='clean_data.pkl', batch_size=64, lr=0.05, test_size=0.2, seed=42):
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        return X_train, X_val, y_train, y_val

    def build_dataloaders(self, X_train, y_train, X_val, y_val):
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2)

    def build_model(self, input_dim):
        model = nn.Sequential()
        model.add_module('linear', nn.Linear(input_dim, 1))
        self.model = model

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self, n_epochs=100):

        X, y = self.load_data()
        X_train, X_val, y_train, y_val = self.process_data(X, y)
        self.build_dataloaders(X_train, y_train, X_val, y_val)
        self.build_model(input_dim=X.shape[1])

        self.arch = Architecture(self.model, self.loss_fn, self.optimizer)
        self.arch.set_loaders(self.train_loader, self.val_loader)
        self.arch.set_seed(self.seed)
        self.arch.train(n_epochs=n_epochs)

        return self.arch, self.scaler
