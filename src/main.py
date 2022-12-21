import argparse
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import numpy as np
import sys
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import load_dataset, shuffle_data
from model import KGHeteroGNN, KGHeteroGNN_Initial, train, test, KGHeteroGNN1


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='FakeNewsNet', help="['FakeNewsNet', 'Covid-19', 'LIAR_PANTS', 'pan2020']")
    
    # GNN related parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.default=200')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Dim of 1st layer GNN. 32,64,128,256, default=256')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers. 2,3, default=2')
    parser.add_argument('--learning_rate', default=0.005, help='Learning rate of the optimiser. 0.0001, 0.001, default=0.005')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay of the optimiser. default=5e-4')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = arg_parser()
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hgraph = load_dataset(args.dataset)
    args.device = device
    hgraph = shuffle_data(hgraph, args)

    for i in range(10):
        # Initialize model parameters
        model = KGHeteroGNN_Initial(hidden_channels=args.hidden_channels, out_channels=2, num_layers=args.gnn_layers)
        
        model.to(device)
        hgraph.to(device)
        # Initialize parameters via lazy initialization
        with torch.no_grad():  # Initialize lazy modules.
            out = model(hgraph.x_dict, hgraph.edge_index_dict)

        train(model, hgraph, args)
        with torch.no_grad():
            test(model, hgraph, args, i)
        
