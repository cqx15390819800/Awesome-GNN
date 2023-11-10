import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TransE


def train(entitys, relation, triples, train_set, epochs=5, batchSize=1024, lr=0.01, dim=128):
    net = TransE(max(entitys) + 1 , max(relation) + 1, dim)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    for e in range(epochs):
        net.train()
        all_lose = 0
        for X in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            optimizer.zero_grad()
            loss = net(X)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose/(len(triples))))
