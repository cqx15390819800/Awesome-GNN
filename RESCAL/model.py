import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class RESCAL(nn.Module):

    def __init__(self, n_entitys, n_relations, dim=128, margin=1):
        super().__init__()
        self.margin = margin        # hinge_loss 中的差距
        self.n_entitys = n_entitys  # 实体的数量
        self.n_relations = n_relations # 关系的数量
        self.dim = dim  # embedding 的长度

        # 随机初始化实体的 embedding
        self.e = nn.Embedding(self.n_entitys, dim, max_norm=1)
        # 随机初始化关系的 embedding
        self.r = nn.Embedding(self.n_relations, dim * dim, max_norm=1)

    def forward(self, X):
        x_pos, x_neg = X
        y_pos = self.predict(x_pos)
        y_neg = self.predict(x_neg)
        return self.hinge_loss(y_pos, y_neg)

    def predict(self, x):
        h, r, t = x
        h = self.e(h)
        r = self.r(r)
        t = self.e(t)
        t = torch.unsqueeze(t, dim = 2)         # [batch_size, dim, 1]
        r = r.reshape(-1, self.dim, self.dim)   # [batch_size, dim, dim]
        tr = torch.matmul(r, t)      # [batch_size, dim, 1]
        tr = torch.squeeze(tr)       # [batch_size, dim]
        score = torch.sum(h*tr, -1)  # [batch_size]
        return -score   # 注意这个符号（语义匹配模型和翻译距离模型的不同）

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))
