import torch
from dgl.nn.pytorch import TransE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TransR(nn.Module):

    def __init__(self, n_entitys, n_relations, k_dim=128, r_dim=64 , margin=1):
        super().__init__()
        self.margin = margin        # hinge_loss 中的差距
        self.n_entitys = n_entitys  # 实体的数量
        self.n_relations = n_relations # 关系的数量
        self.k_dim = k_dim          # 实体 embedding 的长度
        self.r_dim = r_dim          # 关系 embedding 的长度

        # 随机初始化实体的 embedding
        self.e = nn.Embedding(self.n_entitys, k_dim, max_norm=1)
        # 随机初始化关系的 embedding
        self.r = nn.Embedding(self.n_relations, r_dim, max_norm=1)
        # 随机初始化变换矩阵
        self.Mr = nn.Embedding(self.n_relations, k_dim * r_dim, max_norm=1)

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
        mr = self.Mr(r_index)
        score = self.Rtransfer(h, mr) + r - self.Rtransfer(t, mr)
        return torch.sum(score**2, dim=1)**0.5

    def Rtransfer(self, e, mr):
        # [ batch_size, 1, e_dim ]
        e = torch.unsqueeze(e, dim=1)
        # [ batch_size, e_dim, r_dim ]
        mr = mr.reshape(-1, self.k_dim, self.r_dim)
        # [ batch_size, 1, r_dim ]
        result = torch.matmul(e, mr)
        # [ batch_size, r_dim ]
        result = torch.squeeze(result)
        return result

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))
