import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from data import graphsage
from model import GCN4Rec
from sklearn.metrics import precision_score, recall_score, accuracy_score


@torch.no_grad()
def doEva(net, d, g):
    net.eval()
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    # i_index = i.detach().numpy()
    # edges = graphsage(G, i_index)
    out = net(u, i, g)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(g, user_set, train_set, test_set, entitys, epoch=5, batchSize=1024, dim=128, hidden_dim=64, lr=0.01, eva_per_epochs=1):

    model = GCN4Rec(max(user_set) + 1, max(entitys) + 1, dim, hidden_dim)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        all_lose = 0
        for u, i, r in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy())   # r == label  detach 防止梯度影响 r
            optimizer.zero_grad()
            # 因为根据 torch.utils.data.DataLoader 得到一批次 i 是 tensor 类型数据，所以先转成 numpy 类型
            # i_index = i.detach().numpy()
            # GraphSAGE：传入全量图数据 g 与每批次的物品索引得到表示子图的边集
            # edges = graphsage(g, i_index)   # (2, 7299)
            # 传入每批次的用户索引，物品索引，及图采样得到的边集开始前向传播

            logits = model(u, i, g)
            loss = criterion(logits, r)
            all_lose += loss
            loss.backward()
            optimizer.step()

        print('epoch {}, avg_loss = {:.4f}'.format(e, all_lose / (len(train_set) // batchSize)))     # 每个 batch 的平均 loss

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(model, train_set, g)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(model, test_set, g)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))


    return model



