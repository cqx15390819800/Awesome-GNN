import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CKE


def doEva(net, d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net.rec_predict(u, i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc

def train(entitys, relation, triples, kgTrainSet, users, items, train_set, test_set, epochs=10, rec_batchSize=1024, kg_batchSize=512, lr=0.01, dim=128, eva_per_epochs=1):
    # 初始化模型
    net = CKE(max(users) + 1, max(entitys) + 1 , max(relation) + 1, dim)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        net.train()
        all_lose = 0
        # 同时采样用户物品三元组及知识图谱三元组数据, 因计算过程中互相独立所以 batch_size 可设成不一样的值
        for rec_set, kg_set in tqdm(zip(DataLoader(train_set, batch_size=rec_batchSize, shuffle=True),
                       DataLoader(kgTrainSet, batch_size=kg_batchSize, shuffle=True))):
            optimizer.zero_grad()
            loss = net(rec_set, kg_set)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose/(len(train_set))))

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train_set)
            print('train: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
            p, r, acc = doEva(net, test_set)
            print('test: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f}'.format(p, r, acc))
