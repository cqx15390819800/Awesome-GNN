import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from data import graphsage
from model import GFM
from sklearn.metrics import precision_score, recall_score, accuracy_score


def train(g, user_set, train_set, test_set, entitys, epoch=5, batchSize=1024, dim=128, hidden_dim=64, lr=0.01, eva_per_epochs=1):

    model = GFM(max(user_set) + 1, max(entitys) + 1, dim, hidden_dim)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    sampler = dgl.dataloading.NeighborSampler([5, 25])  # 15 -> 10 -> 5 -> 1      这个采样不适用于 GFM，这样使得邻边太少
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # block0 18129  13652   4620
    # block1 13652  10258   3286
    # block2 10258  1024    8990
    for e in range(epoch):

        # train
        model.train()
        for u, i, r in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):     # 数据采样
            graph_dataloader = dgl.dataloading.DataLoader(g, i, sampler, batch_size=len(i), shuffle=True, drop_last=False, num_workers=4)
            for input_nodes, output_nodes, blocks in graph_dataloader:  # batch_size=len(i) 相当于获取 i 的子图 blocks

                r = torch.FloatTensor(r.detach().numpy())   # r == label  detach 防止梯度影响 r
                optimizer.zero_grad()

                logits = model(u, i, blocks)

                loss = criterion(logits, r)
                loss.backward()
                optimizer.step()

                y_pred = np.array([1 if i >= 0.5 else 0 for i in logits])
                y_true = r.detach().numpy()
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                acc = accuracy_score(y_true, y_pred)
                print('epoch {}, loss = {:.4f}'.format(e, loss))  # 每个 batch 的 loss
                print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(precision, recall, acc))

        # test
        model.eval()
        for u, i, r in tqdm(DataLoader(test_set, batch_size=batchSize, shuffle=True)):
            graph_dataloader = dgl.dataloading.DataLoader(g, i, sampler, batch_size=len(i), shuffle=True, drop_last=False, num_workers=4)
            for input_nodes, output_nodes, blocks in graph_dataloader:
                r = torch.FloatTensor(r.detach().numpy())   # r == label  detach 防止梯度影响 r
                optimizer.zero_grad()

                logits = model(u, i, blocks)

                y_pred = np.array([1 if i >= 0.5 else 0 for i in logits])
                y_true = r.detach().numpy()
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                acc = accuracy_score(y_true, y_pred)
                print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(precision, recall, acc))

    return model



