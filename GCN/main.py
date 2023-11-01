import argparse
import torch

import dgl

from data import get_data
from model import GCN
from utils import train, evaluate

if __name__ == "__main__":
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()

    print(f"Training with DGL built-in GraphConv module.")

    # data
    data = get_data(args)
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # feature
    degs = g.out_degrees().float()
    norm = torch.pow(degs, -0.5)    # 1 / d ^ (1/2)
    norm[torch.isinf(norm)] = 0     # 某个节点可能没有出边，这一步很重要
    g.ndata['out_degree_norm'] = norm.unsqueeze(1)      # (node_num, 1)

    # GCN
    in_features_dim = features.shape[1]
    hid_features_dim = 16
    out_features_dim = data.num_classes
    model = GCN(g, in_features_dim, hid_features_dim, out_features_dim, n_layers=1, dropout=0.5).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # train
    print("Training...")
    train(g, features, labels, masks, model)

    # test
    print("Testing...")     # Test accuracy 0.8100
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))