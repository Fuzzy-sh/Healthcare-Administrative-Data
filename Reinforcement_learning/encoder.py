import os
import datetime
import logging
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from encoder import Autoencoder
from utils import TrainAutoencoder
from dataloader import DataLoader

# Device configuration
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(3)


# Main Training Loop
if __name__ == "__main__":
    best_overall_loss = float("inf")
    best_latent_dim = None
    # Constants
    BATCH_SIZE = 128
    NUM_EPOCHS = 3000
    PATIENCE = 10
    MIN_EPOCHS = 100
    # Logging setup
    logging = logging.basicConfig(
        filename="autoencoder_training_logs.log",
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info = logging.info(f"Program started at {datetime.datetime.now()}")

    processed_data_path = tc.processed_data_path

    # Load data
    dl = DataLoader()

    X_train = dl.load_data(processed_data_path, "train")
    X_val = dl.load_data(processed_data_path, "validation")
    X_test = dl.load_data(processed_data_path, "test")

    train_loader = dl.create_dataloader(X_train, BATCH_SIZE)
    val_loader = dl.create_dataloader(X_val, BATCH_SIZE, shuffle=False)

    logging.info(
        f"Data shapes - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}"
    )

    # Train the encoder
    encoder = Autoencoder(X_train.shape[1], 64, 3, nn.ReLU).to(device)
    te = TrainAutoencoder(
        X_train,
        train_loader,
        val_loader,
        device,
        BATCH_SIZE,
        NUM_EPOCHS,
        PATIENCE,
        MIN_EPOCHS,
    )
    # error = te.train_encoder(X_train, train_loader, val_loader, device, BATCH_SIZE, NUM_EPOCHS, PATIENCE, MIN_EPOCHS)

    for latent_dim in [64]:  # Extendable to multiple latent dimensions
        for num_hidden_layers in range(5):  # Testing different numbers of hidden layers
            for activation_fn in [nn.ReLU, nn.LeakyReLU]:
                best_loss = te.train_autoencoder(
                    train_loader,
                    val_loader,
                    latent_dim,
                    num_hidden_layers,
                    activation_fn,
                )
                logging.info(
                    f"Latent dim: {latent_dim}, Layers: {num_hidden_layers}, Activation: {activation_fn.__name__}, Loss: {best_loss}"
                )
                if best_loss < best_overall_loss:
                    best_overall_loss = best_loss
                    best_latent_dim = latent_dim

    # Save the best encoder
    best_model = Autoencoder(
        X_train.shape[1], best_latent_dim, num_hidden_layers, nn.ReLU
    )
    best_model.load_state_dict(
        torch.load(
            os.path.join(
                tc.models_path,
                f"autoencoder_sigmoid_{best_latent_dim}",
                "checkpoint.pth",
            )
        )["state_dict"]
    )
    encoder_path = os.path.join(tc.models_path, "best_encoder")
    os.makedirs(encoder_path, exist_ok=True)
    torch.save(
        best_model.encoder.state_dict(), os.path.join(encoder_path, "encoder.pth")
    )
    logging.info(
        f"Best encoder saved with latent dim {best_latent_dim} and loss {best_overall_loss:.6f}"
    )
