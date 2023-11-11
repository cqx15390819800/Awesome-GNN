import torch
from dgl.nn.pytorch import TransE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TransH(nn.Module):

    def __init__(self, n_entitys, n_relations, dim=128, margin=1):
        super().__init__()
        self.margin = margin        # hinge_loss 中的差距
        self.n_entitys = n_entitys  # 实体的数量
        self.n_relations = n_relations # 关系的数量
        self.dim = dim  # embedding 的长度

        # 随机初始化实体的 embedding
        self.e = nn.Embedding(self.n_entitys, dim, max_norm=1)
        # 随机初始化关系的 embedding
        self.r = nn.Embedding(self.n_relations, dim, max_norm=1)
        # 随机初始化法向量的 embedding
        self.wr = nn.Embedding(self.n_relations, dim, max_norm=1)    # max_norm = 1 将该法向量规范为单位法向量


    def forward(self, X):
        x_pos, x_neg = X
        y_pos = self.predict(x_pos)
        y_neg = self.predict(x_neg)
        return self.hinge_loss(y_pos, y_neg)

    def predict(self, x):
        h, r_index, t = x
        h = self.e(h)
        r = self.r(r_index)
        t = self.e(t)
        wr = self.wr(r_index)   # 当前关系的法向量
        score = self.Htransfer(h, wr) + r - self.Htransfer(t, wr)   # score = h' + r - t'
        return torch.sum(score**2, dim=1)**0.5  # L2 范数

    def Htransfer(self, e, wr): # h_wr
        return e - torch.sum(e * wr, dim=1, keepdim=True) * wr

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))
