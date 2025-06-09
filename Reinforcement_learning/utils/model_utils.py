import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

import logging
import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataloader import DataLoaderClass

# torch.cuda.set_device(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn


class AutoencoderGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_hidden_layers,
        hidden_size,
        activation_fn=nn.ReLU,
    ):
        """
        Autoencoder using GRU layers.

        Args:
            input_dim (int): Number of input features.
            latent_dim (int): Size of the latent representation.
            num_hidden_layers (int): Number of GRU layers in the encoder and decoder.
            hidden_size (int): Size of the GRU hidden state.
            activation_fn (torch.nn.Module): Activation function (default: nn.ReLU).
        """
        super(AutoencoderGRU, self).__init__()

        # Encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_hidden_layers,
            batch_first=True,
        )
        self.latent_fc = nn.Linear(
            hidden_size, latent_dim
        )  # Map GRU output to latent space

        # Decoder GRU
        self.latent_to_hidden = nn.Linear(
            latent_dim, hidden_size
        )  # Map latent space to GRU hidden size
        self.decoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_hidden_layers,
            batch_first=True,
        )
        self.output_fc = nn.Linear(
            hidden_size, input_dim
        )  # Map GRU output back to input space

        # Activation
        self.activation = activation_fn()

    def forward(self, x):
        # Encoder: GRU -> Fully Connected (latent space)
        _, h_n = self.encoder_gru(
            x
        )  # h_n contains the hidden state from the last time step
        latent = self.activation(
            self.latent_fc(h_n[-1])
        )  # Use the last layer's hidden state

        # Decoder: Fully Connected (latent to hidden) -> GRU -> Fully Connected (output)
        h_0 = (
            self.latent_to_hidden(latent)
            .unsqueeze(0)
            .repeat(self.decoder_gru.num_layers, 1, 1)
        )
        output, _ = self.decoder_gru(x, h_0)
        reconstructed = self.output_fc(output)

        return reconstructed

    def encode(self, x):
        # Encoder only
        _, h_n = self.encoder_gru(x)
        latent = self.activation(self.latent_fc(h_n[-1]))
        return latent

    def decode(self, latent, sequence_length):
        # Decoder only
        h_0 = (
            self.latent_to_hidden(latent)
            .unsqueeze(0)
            .repeat(self.decoder_gru.num_layers, 1, 1)
        )
        decoder_input = torch.zeros(
            (latent.size(0), sequence_length, latent.size(1)), device=latent.device
        )
        output, _ = self.decoder_gru(decoder_input, h_0)
        reconstructed = self.output_fc(output)
        return reconstructed


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_hidden_layers, activation_fn):
        super().__init__()
        # Encoder
        encoder_layers = [
            nn.Linear(input_dim, latent_dim * 2**num_hidden_layers),
            activation_fn(),
        ]
        for i in range(num_hidden_layers, 0, -1):
            encoder_layers.append(
                nn.Linear(latent_dim * 2**i, latent_dim * 2 ** (i - 1))
            )
            encoder_layers.append(nn.Sigmoid() if i == 1 else activation_fn())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(num_hidden_layers):
            decoder_layers.append(
                nn.Linear(latent_dim * 2**i, latent_dim * 2 ** (i + 1))
            )
            decoder_layers.append(activation_fn())
        decoder_layers.append(nn.Linear(latent_dim * 2**num_hidden_layers, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


class TrainAutoencoder:
    def __init__(self, NUM_EPOCHS, device, MIN_EPOCHS, PATIENCE):

        self.device = device
        self.NUM_EPOCHS = NUM_EPOCHS
        self.MIN_EPOCHS = MIN_EPOCHS
        self.PATIENCE = PATIENCE

    # Training Function
    def train_autoencoder(
        self,
        train_loader,
        val_loader,
        latent_dim,
        num_hidden_layers,
        activation_fn,
        models_path,
        X_train,
        prefix,
        best_loss=float("inf"),
    ):
        model = Autoencoder(
            X_train.shape[1], latent_dim, num_hidden_layers, activation_fn
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        patience_counter = 0
        autoencoder_path = os.path.join(
            models_path, f"{prefix}autoencoder_sigmoid_{latent_dim}"
        )
        os.makedirs(autoencoder_path, exist_ok=True)

        for epoch in range(self.NUM_EPOCHS):
            model.train()
            print("Epoch:", epoch)
            for X_batch, _ in tqdm(train_loader):
                # X_batch = X_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), X_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            # predicted =model(X_batch) #model(X_batch.to(self.device))
            # true_value = X_batch #X_batch.to(self.device)
            val_loss = sum(
                criterion(model(X_batch), X_batch).item() for X_batch, _ in val_loader
            ) / len(val_loader)

            logging.info(f"Epoch {epoch}, Val Loss: {val_loss:.6f}")
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "parameters": {
                        "input_dim": X_train.shape[1],
                        "latent_dim": latent_dim,
                        "num_hidden_layers": num_hidden_layers,
                        "activation_fn": activation_fn,
                    },
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }
                torch.save(checkpoint, os.path.join(autoencoder_path, "checkpoint.pth"))
                logging.info(
                    f"Model for {prefix} saved at epoch {epoch} with loss {val_loss:.6f}; num_hidden_layers : {num_hidden_layers}, latent_dim : {latent_dim}, activation_fn : {activation_fn}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if epoch >= self.MIN_EPOCHS and patience_counter > self.PATIENCE:
                    logging.info(
                        f"Early stopping for {prefix} at epoch {epoch} for latent dim {latent_dim}; num_hidden_layers : {num_hidden_layers}, activation_fn : {activation_fn}"
                    )
                    break

        return best_loss

    def call(self, train_loader, val_loader, X_train, models_path, latent_dim, prefix):
        best_overall_loss = float("inf")
        best_latent_dim = None
        filename = f"{prefix}autoencoder_training_logs_{latent_dim}.log"
        logging.basicConfig(
            filename=filename,  # Set the log file name
            level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logging.info(f"Start the program at {datetime.datetime.now()}")
        for latent_dim in [latent_dim]:  # [8, 16, 32, 64, 128, 225]:
            best_loss_for_latent_dim = float("inf")
            for num_hidden_layers in [0, 1, 2, 3, 4]:
                for activation_fn in [nn.ReLU, nn.LeakyReLU]:
                    loss = self.train_autoencoder(
                        train_loader,
                        val_loader,
                        latent_dim,
                        num_hidden_layers,
                        activation_fn,
                        models_path,
                        X_train,
                        prefix,
                        best_loss_for_latent_dim,
                    )
                    logging.info(
                        f"Prefix: {prefix}, Latent dim: {latent_dim}, Num layers: {num_hidden_layers}, Activation: {activation_fn}, Loss: {loss}"
                    )
                    if loss < best_overall_loss:
                        best_overall_loss = loss
                        best_latent_dim = latent_dim
                        best_num_hidden_layers = num_hidden_layers
                        best_activation_fn = activation_fn
                        best_loss_for_latent_dim = loss
                        logging.info(
                            f"Best loss for {prefix} so far: {best_overall_loss} for latent dimension {best_latent_dim} with number of hidden layers: {best_num_hidden_layers} and best activation function is {best_activation_fn} .  Time: {datetime.datetime.now()}"
                        )

        # Loading the best model overall
        # best_model = Autoencoder(X_train.shape[1], best_latent_dim, num_hidden_layers, activation_fn)
        # best_model.load_state_dict(torch.load(os.path.join(models_path, f'autoencoder_sigmoid_{best_latent_dim}', 'checkpoint.pth'))['state_dict'])
        # encoder_model = best_model.encoder
        # best_model_path = os.path.join(models_path, 'best_encoder')
        # os.makedirs(best_model_path, exist_ok=True)
        # torch.save(encoder_model.state_dict(), os.path.join(best_model_path, 'encoder.pth'))

        print(
            f"Best model for {prefix}with latent dimension {best_latent_dim}, loss {best_overall_loss}, with number of hidden layers: {best_num_hidden_layers} and best activation function is {best_activation_fn}."
        )
        logging.info(
            f"Best model for {prefix} with latent dimension {best_latent_dim}, loss {best_overall_loss}, with number of hidden layers: {best_num_hidden_layers} and best activation function is {best_activation_fn}."
        )

        # logging.info(f"Best model with latent dimension {best_latent_dim} and loss {best_overall_loss} saved.")

        logging.info(f"End the program at {datetime.datetime.now()}")


class LoadEncoder:
    def __init__(self, param_dic):
        self.param_dic = param_dic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dl = DataLoaderClass(param_dic)

    def return_data_encoder(self, train_path, latent_dims, prefixes, model_path):
        train_encoded_df = {}

        for prefix, latent_dim in zip(prefixes, latent_dims):

            autoencoder_name = f"{prefix}autoencoder_sigmoid_{latent_dim}"
            X_ = self.dl.load_data(train_path, prefix, self.device)
            autoencoder_model_path = os.path.join(
                model_path, autoencoder_name, "checkpoint.pth"
            )
            checkpoint = torch.load(autoencoder_model_path, weights_only=False)
            params = checkpoint["parameters"]
            autoencoder = Autoencoder(
                params["input_dim"],
                params["latent_dim"],
                params["num_hidden_layers"],
                params["activation_fn"],
            )
            autoencoder.load_state_dict(checkpoint["state_dict"])

            autoencoder = autoencoder.to(self.device)
            autoencoder.eval()
            print(f"Loaded the autoencoder model from {autoencoder_model_path}")
            train_encoded = autoencoder.encode(X_).detach().cpu().numpy()
            train_encoded_df[prefix] = pd.DataFrame(
                train_encoded,
                columns=[f"{prefix}{i}" for i in range(train_encoded.shape[1])],
            )
            # val_encoded_df[prefix] = pd.DataFrame(val_encoded, columns=[f'{prefix}{i}' for i in range(val_encoded.shape[1])])
        data = pd.concat([train_encoded_df[prefix] for prefix in prefixes], axis=1)
        # print (data.columns)
        print(
            f"Finished encoding the features using the autoencoder for {prefix}- latent dim = {latent_dim}"
        )
        return data

    def return_encod_obs(self, train_path, latent_dim, prefix, model_path):
        X_ = self.dl.load_data(train_path, prefix, self.device)
        autoencoder_name = f"{prefix}autoencoder_sigmoid_{latent_dim}"
        autoencoder_model_path = os.path.join(
            model_path, autoencoder_name, "checkpoint.pth"
        )
        checkpoint = torch.load(autoencoder_model_path, weights_only=False)
        params = checkpoint["parameters"]
        autoencoder = Autoencoder(
            params["input_dim"],
            params["latent_dim"],
            params["num_hidden_layers"],
            params["activation_fn"],
        )
        autoencoder.load_state_dict(checkpoint["state_dict"])
        # print(autoencoder.named_parameters())
        # for name, param in autoencoder.named_parameters():
        #     print(name, param, param.size())
        autoencoder = autoencoder.to(self.device)
        autoencoder.eval()

        print(f"Loaded the autoencoder model from {autoencoder_model_path}")
        train_encoded_ = autoencoder.encode(X_).detach().cpu().numpy()
        train_encoded_df = pd.DataFrame(
            train_encoded_,
            columns=[f"{prefix}{i}" for i in range(train_encoded_.shape[1])],
        )
        # val_encoded_df[prefix] = pd.DataFrame(val_encoded, columns=[f'{prefix}{i}' for i in range(val_encoded.shape[1])])

        # print (data.columns)
        print(
            f"Finished encoding the features using the autoencoder for {prefix}- latent dim = {latent_dim}"
        )
        return train_encoded_df


class TrainAutoencoderGRU:
    def __init__(self, NUM_EPOCHS, device, MIN_EPOCHS, PATIENCE):
        self.device = device
        self.NUM_EPOCHS = NUM_EPOCHS
        self.MIN_EPOCHS = MIN_EPOCHS
        self.PATIENCE = PATIENCE

    # Training Function
    def train_autoencoder(
        self,
        train_loader,
        val_loader,
        latent_dim,
        num_hidden_layers,
        hidden_size,
        models_path,
        X_train,
        prefix,
        best_loss=float("inf"),
    ):
        print(print("X_train shape:", X_train.shape))
        sequence_length, input_dim = X_train.shape[1], X_train.shape[2]
        model = AutoencoderGRU(
            input_dim, latent_dim, num_hidden_layers, hidden_size
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        patience_counter = 0
        autoencoder_path = os.path.join(
            models_path, f"{prefix}autoencoder_gru_{latent_dim}"
        )
        os.makedirs(autoencoder_path, exist_ok=True)

        for epoch in range(self.NUM_EPOCHS):
            model.train()
            print("Epoch:", epoch)
            for X_batch, _ in tqdm(train_loader):
                X_batch = X_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), X_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion(
                        model(X_batch.to(self.device)), X_batch.to(self.device)
                    ).item()
                    for X_batch, _ in val_loader
                ) / len(val_loader)

            logging.info(f"Epoch {epoch}, Val Loss: {val_loss:.6f}")
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "parameters": {
                        "input_dim": input_dim,
                        "latent_dim": latent_dim,
                        "num_hidden_layers": num_hidden_layers,
                        "hidden_size": hidden_size,
                    },
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }
                torch.save(checkpoint, os.path.join(autoencoder_path, "checkpoint.pth"))
                logging.info(
                    f"Model for {prefix} saved at epoch {epoch} with loss {val_loss:.6f}."
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if epoch >= self.MIN_EPOCHS and patience_counter > self.PATIENCE:
                    logging.info(
                        f"Early stopping for {prefix} at epoch {epoch} for latent dim {latent_dim}."
                    )
                    break

        return best_loss

    def call(self, train_loader, val_loader, X_train, models_path, latent_dim, prefix):
        best_overall_loss = float("inf")
        best_latent_dim = None
        filename = f"{prefix}autoencoder_training_logs_{latent_dim}.log"
        logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logging.info(f"Start the program at {datetime.datetime.now()}")
        for latent_dim in [latent_dim]:
            best_loss_for_latent_dim = float("inf")
            for num_hidden_layers in [1, 2, 3]:
                for hidden_size in [16, 32, 64]:
                    loss = self.train_autoencoder(
                        train_loader,
                        val_loader,
                        latent_dim,
                        num_hidden_layers,
                        hidden_size,
                        models_path,
                        X_train,
                        prefix,
                        best_loss_for_latent_dim,
                    )
                    logging.info(
                        f"Prefix: {prefix}, Latent dim: {latent_dim}, Num layers: {num_hidden_layers}, Hidden size: {hidden_size}, Loss: {loss}"
                    )
                    if loss < best_overall_loss:
                        best_overall_loss = loss
                        best_latent_dim = latent_dim
                        best_num_hidden_layers = num_hidden_layers
                        best_hidden_size = hidden_size
                        best_loss_for_latent_dim = loss
                        logging.info(
                            f"Best loss for {prefix} so far: {best_overall_loss} for latent dimension {best_latent_dim}."
                        )

        print(
            f"Best model for {prefix} with latent dimension {best_latent_dim}, loss {best_overall_loss}, with num_hidden_layers: {best_num_hidden_layers}, hidden_size: {best_hidden_size}."
        )
        logging.info(f"End the program at {datetime.datetime.now()}")


class LoadEncoderGRU:
    def __init__(self, param_dic):
        self.param_dic = param_dic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dl = DataLoaderClass(param_dic)

    def return_data_encoder(self, train_path, latent_dims, prefixes, model_path):
        train_encoded_df = {}
        for prefix, latent_dim in zip(prefixes, latent_dims):
            autoencoder_name = f"{prefix}autoencoder_gru_{latent_dim}"
            X_ = self.dl.load_data(train_path, prefix, self.device)
            autoencoder_model_path = os.path.join(
                model_path, autoencoder_name, "checkpoint.pth"
            )
            checkpoint = torch.load(autoencoder_model_path, map_location=self.device)
            params = checkpoint["parameters"]
            autoencoder = AutoencoderGRU(
                params["input_dim"],
                params["latent_dim"],
                params["num_hidden_layers"],
                params["hidden_size"],
            )
            autoencoder.load_state_dict(checkpoint["state_dict"])
            autoencoder = autoencoder.to(self.device)
            autoencoder.eval()
            print(f"Loaded the autoencoder model from {autoencoder_model_path}")
            train_encoded = autoencoder.encode(X_).detach().cpu().numpy()
            train_encoded_df[prefix] = pd.DataFrame(
                train_encoded,
                columns=[f"{prefix}{i}" for i in range(train_encoded.shape[1])],
            )
        data = pd.concat([train_encoded_df[prefix] for prefix in prefixes], axis=1)
        print(f"Finished encoding the features using the autoencoder for {prefix}.")
        return data
