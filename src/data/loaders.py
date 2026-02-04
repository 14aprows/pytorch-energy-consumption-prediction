from torch.utils.data import DataLoader
from .dataset import EnergyDataset
from sklearn.model_selection import train_test_split

def create_loaders(X, y, batch_size=128, val_size=0.2, seed=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=seed)

    train_ds = EnergyDataset(X_train, y_train)
    val_ds = EnergyDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader