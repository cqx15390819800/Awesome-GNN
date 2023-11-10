import torch
from dgl.nn.pytorch import TransE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TransE( nn.Module ):

    def __init__(self, n_entitys, n_relations, dim=128, margin=1):
        super().__init__()
        self.margin = margin            # hinge_loss 中的差距 m
        self.n_entitys = n_entitys      # 实体的数量
        self.n_relations = n_relations  # 关系的数量
        self.dim = dim                  # embedding 的长度

        # 随机初始化实体的 embedding
        self.e = nn.Embedding(self.n_entitys, dim, max_norm=1)   # head tail 实体肯定是在同一向量空间的
        # 随机初始化关系的 embedding
        self.r = nn.Embedding(self.n_relations, dim, max_norm=1)

    def forward(self, X):
        x_pos, x_neg = X
        y_pos = self.predict(x_pos)
        y_neg = self.predict(x_neg)
        return self.hinge_loss(y_pos, y_neg)  # TransE 返回的是 loss，而不是 logits，这个 loss 不是核心，核心是实体和关系的 embedding

    def predict(self, x):
        h, r, t = x
        h = self.e(h)
        r = self.r(r)
        t = self.e(t)
        score = h + r - t
        return torch.sum(score**2, dim=1)**0.5

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))   # max(0, dis) -> loss_sum
