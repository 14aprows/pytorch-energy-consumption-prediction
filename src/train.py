import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.data.loaders import create_loaders
from src.models.mlp import MLPRegressor
from src.trainers.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))

def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df =  pd .read_csv("dataset/Energy_consumption_dataset.csv")

    target = "EnergyConsumption"

    categorical_cols = [
        "DayOfWeek",
        "Holiday",
        "HVACUsage",
        "LightingUsage",
    ]

    numerical_cols = [
        "Month",
        "Hour",
        "Temperature",
        "Humidity",
        "SquareFootage",
        "Occupancy",
        "RenewableEnergy",
    ]

    X_cat = df[categorical_cols]
    X_num = df[numerical_cols]

    cat_encoder = OrdinalEncoder()
    X_cat_enc = cat_encoder.fit_transform(X_cat)

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    X = np.concatenate([X_cat_enc, X_num_scaled], axis=1)
    y = np.log1p((df[target]).values).astype(np.float32)

    train_loader, val_loader = create_loaders(X, y, batch_size=128, val_size=0.2, seed=42)

    model = MLPRegressor(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=rmse,
        device=device,
        epochs=80,
        save_path="checkpoints/best_mlp_model.pth"
    )

    trainer.fit(train_loader, val_loader)

train()