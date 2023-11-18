import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class CKE(nn.Module):

    def __init__(self, n_users, n_entitys, n_relations, e_dim = 128, margin = 1, alpha=0.2):
        super().__init__()
        self.margin = margin
        self.u_emb = nn.Embedding(n_users, e_dim)       # 用户向量
        self.e_emb = nn.Embedding(n_entitys, e_dim)     # 实体向量
        self.r_emb = nn.Embedding(n_relations, e_dim)   # 关系向量

        self.BCEloss = nn.BCELoss()

        self.alpha = alpha  # kge 损失函数的计算权重

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))

    # kge 采用最基础的 TransE 算法
    def kg_predict(self, x):
        h, r, t = x
        h = self.e_emb(h)
        r = self.r_emb(r)
        t = self.e_emb(t)
        score = h + r - t
        return torch.sum(score**2, dim=1)**0.5

    # 计算 kge 损失函数
    def calculatingKgeLoss(self, kg_set):
        x_pos, x_neg = kg_set
        y_pos = self.kg_predict(x_pos)
        y_neg = self.kg_predict(x_neg)
        return self.hinge_loss(y_pos, y_neg)

    # 双塔
    def rec_predict(self, u, i):
        u = self.u_emb(u)
        i = self.e_emb(i)
        y = torch.sigmoid(torch.sum(u*i, dim=1))
        return y

    # 计算推荐损失函数
    def calculatingRecLoss(self, rec_set):
        u, i, y = rec_set
        y_pred = self.rec_predict(u, i)
        y = torch.FloatTensor(y.detach().numpy())   # 为了避免梯度影响到 label，所以进行 detach 操作
        return self.BCEloss(y_pred, y)

    # 前向传播
    def forward(self, rec_set, kg_set):
        rec_loss = self.calculatingRecLoss(rec_set)
        kg_loss = self.calculatingKgeLoss(kg_set)
        # 分别得到推荐产生的损失函数与 kge 产生的损失函数加权相加后返回
        return rec_loss + self.alpha * kg_loss
