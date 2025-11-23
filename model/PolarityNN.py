import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import TensorDataset, DataLoader
import datetime
import random
import string
import os
from tqdm import tqdm
import numpy as np

class PolarityNN(nn.Module):
    """
        this model is for film review to predict if the review is good or bad

    """
    def __init__(self, input_size=5000, hidden_size=128):
        super(PolarityNN, self).__init__()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        self.model_name = f"PolarityNN_{timestamp}_{rand_suffix}" 
        
        # neural layer
        self.model = nn.Sequential(
            # layer: 5000 -> 128
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # layer: 128 -> 64
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # output layer: 64 -> 1 (classification binaire)
            nn.Linear(64, 1),
            nn.Sigmoid()  
        )
        
    def forward(self, x):
        # x => (batch_size, 5000)
        return self.model(x)


    def predict(self, x):
        with torch.no_grad():
            probs = self.forward(x)
            return (probs > 0.5).float()
        
    def score(self, _, val_loader):
        total = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self(batch_x)

                predicted = (outputs > 0.5).float()
                correct = np.mean(predicted.view(-1) == batch_y)
                total.append(correct)

        return np.mean(total)
            
    def fit(self, train_loader, val_loader, num_epochs:int=10, learning_rate:float=0.001, 
                         optimizer:str="adam", criterion:str="bceloss",optim_params=None, criterion_params=None):
        
        methode_criterion={
            "bceloss":nn.BCELoss,
            "mseloss":nn.MSELoss,
            "l1loss":nn.L1Loss,
            "crossentropyloss":nn.CrossEntropyLoss
        }
        if criterion not in methode_criterion:
            criterion=methode_criterion["bceloss"]()
        else:
            criterion = methode_criterion[criterion](**criterion_params)

        methode_optimizer={
            "adam":opt.Adam,
            "adamax":opt.Adamax,
            "sgd":opt.SGD
        }
        if optimizer not in methode_optimizer:
            optimizer = opt.Adam(self.parameters(), lr=learning_rate)
        else:
            optimizer = methode_optimizer[optimizer](self.parameters(), lr=learning_rate, **optim_params)
        
        train_losses = []
        val_losses = []
        
        progress =tqdm(range(num_epochs*len(train_loader)))
        for epoch in range(num_epochs):
            # torch training mode
            self.train()
            total_train_loss = 0
            
            for batch_x, batch_y in train_loader:
                # Forward pass
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y.float().view(-1, 1))
                
                # Backward pass and optim
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                progress.update(1)
                progress.refresh()
            tqdm.write(f"epoch[{epoch}/{num_epochs}] : loss : {total_train_loss/len(train_loader)}")
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # torch eval
            self.eval()
            total_val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self(batch_x)
                    loss = criterion(outputs, batch_y.float().view(-1, 1))
                    total_val_loss += loss.item()

                    predicted = (outputs > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted.view(-1) == batch_y).sum().item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            accuracy = 100 * correct / total
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}')
            print(f'Val Accuracy: {accuracy:.2f}%\n')
        
        return train_losses, val_losses
    
    def save(self, folder_name="trained"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, folder_name)
        
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_name}.pt")
        
        torch.save(self.state_dict(), path)
        print(f"Model saved at: {path}")
        return path