from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report
from utils import PreprocessData, Encoder, UpdateData, TrainAutoencoder
from  config.config_param import config_param
import argparse
import datetime
import torch 
from torch.utils.data import DataLoader, TensorDataset

import logging
from dataloader import DataLoaderClass
import mdptoolbox
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.cluster import KMeans
from utils import PolicyResolver, WeightedImportanceSampling


def main():
    
    # experiment_name = f'{experiment_type}-{obs_n_clusters}-{act_n_clusters}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    BATCH_SIZE = 128
    # autoencoder loading
    config_params = config_param()
    param_dic = config_params.get_params_updated()
    dl = DataLoaderClass(param_dic)
    pred = PreprocessData(param_dic)
    parser = argparse.ArgumentParser(description='Run the encoder using the given parameters')
    parser.add_argument('--n_dim', type=int, default=64, help='The number of dimensions for the latent space')
    parser.add_argument('--prefix', type=str, default='observation_prefix', help='The number of dimensions for the latent space')
    # action_prefix
    args = parser.parse_args()
    n_dim = args.n_dim
    model_path = param_dic['models_path']
    prefix = param_dic[args.prefix]
    print("Loading the train, test, and validation data")
    train_path ,test_path, val_path = param_dic['split_files']['train'], param_dic['split_files']['test'], param_dic['split_files']['val']
    # train_data, test_data, val_data = pred.read_data(train_path), pred.read_data(test_path), pred.read_data(val_path)
    print("Data loaded successfully!")
    X_train, X_test, X_val = dl.load_data(train_path, prefix, device ), dl.load_data(test_path, prefix, device ),dl.load_data(val_path, prefix , device)
    
    
    print ("Trian the model and save it first")
    train_loader, val_loader = dl.create_dataloader( X_train, batch_size =BATCH_SIZE, shuffle=True), dl.create_dataloader( X_val, batch_size=1, shuffle=True)
    train_autoencoder = TrainAutoencoder(NUM_EPOCHS=100,device=device,MIN_EPOCHS=20,PATIENCE=20)
    train_autoencoder.call(train_loader, val_loader, X_train, model_path, n_dim, prefix)
    


if __name__ == '__main__':

    main()
 