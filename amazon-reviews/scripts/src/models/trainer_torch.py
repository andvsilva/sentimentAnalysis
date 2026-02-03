# models/trainer_torch.py
import torch
import torch.nn as nn

class TorchTrainer:

    def __init__(self, model, lr=1e-3, device="cpu"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        train_loader,
        val_loader=None,
        epochs=10
    ):
        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device).float().view(-1, 1)

                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            history["loss"].append(running_loss / len(train_loader))

            if val_loader:
                val_loss = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {history['loss'][-1]:.4f}")

        return history

    def evaluate(self, loader):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device).float().view(-1, 1)

                outputs = self.model(X)
                loss += self.criterion(outputs, y).item()

        return loss / len(loader)
