import torch 
import os 

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, epochs, save_path):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.save_path = save_path

        self.best_val = float('inf')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def fit(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)

            if val_loss < self.best_val:
                self.best_val = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                flag = " (saved)"
            else:
                flag = ""
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}{flag}")


    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item() 

        return total_loss / len(loader)
    
    def _val_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                total_loss += loss.item() 

        return total_loss / len(loader)