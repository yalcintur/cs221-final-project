import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.autoencoder import Autoencoder
import numpy as np
import wandb


def train_autoencoder(
    train_data, val_data, input_dim, latent_dim, batch_size, epochs=1000, patience=20, device="cuda:2"
):
    wandb.init(
        project="autoencoder-training",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "learning_rate": 2e-3,
            "patience": patience,
            "device": device,
        },
        reinit=True,
    )

    train_data_gpu = torch.tensor(train_data, dtype=torch.float32).to(device)
    val_data_gpu = torch.tensor(val_data, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(train_data_gpu),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data_gpu),
        batch_size=batch_size,
        shuffle=False
    )

    autoencoder = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=2e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    criterion = torch.nn.MSELoss()

    best_val_loss = np.inf
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        autoencoder.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0]  # Data is already on GPU
            _, reconstruction = autoencoder(inputs)
            loss = criterion(reconstruction, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]  # Data is already on GPU
                _, reconstruction = autoencoder(inputs)
                loss = criterion(reconstruction, inputs)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Learning Rate": current_lr,
            "Epoch": epoch + 1
        })

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {current_lr:.5f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), "best_autoencoder.pth")
            wandb.run.summary["best_val_loss"] = best_val_loss
            print(f"Validation loss improved. Saving model.", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}", flush=True)

        if patience_counter >= patience:
            print("Early stopping triggered.", flush=True)
            break

    wandb.finish()
    return autoencoder
