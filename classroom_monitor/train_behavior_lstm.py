"""
Training Script for BehaviorLSTM

Usage:
1. Collect data: The system can be set to log normalized pose vectors (38 dims) along with manual labels.
   You should save these to a CSV or NumPy array.
   Shape per sample should be (sequence_length, 38).
2. Run this script to train the LSTM.
3. It will save `behavior_lstm_weights.pth` which is automatically picked up by `behavior_lstm.py`.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from classroom_monitor.behavior_lstm import BehaviorLSTM

class BehaviorDataset(Dataset):
    def __init__(self, data_path):
        # Expected data format: a dictionary or list of (sequence, label)
        # sequence shape: (16, 38)
        # label shape: (1,) integer
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True).item()
            self.X = torch.FloatTensor(data['X'])
            self.y = torch.LongTensor(data['y'])
        else:
            print(f"Warning: Data file {data_path} not found. Generating dummy data for testing.")
            self.X = torch.randn(100, 16, 38)
            self.y = torch.randint(0, 5, (100,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    dataset_path = 'behavior_dataset.npy'
    dataset = BehaviorDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = BehaviorLSTM(input_dim=38, hidden_dim=64, num_layers=2, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        acc = 100 * correct / total
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {acc:.2f}%")
            
    # Save weights
    os.makedirs('model_weights', exist_ok=True)
    save_path = os.path.join('model_weights', 'behavior_lstm_weights.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")

if __name__ == '__main__':
    train()
