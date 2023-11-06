import dgl.nn as dglnn
import torch
from dgl.nn.pytorch import SAGEConv
from torch import nn
import torch.nn.functional as F


class GCN4Rec(torch.nn.Module):

    def __init__(self, n_users, n_entitys, dim, hidden_dim):
        '''
        :param n_users: 用户数量
        :param n_entitys: 实体数量(物品+物品特征)
        :param dim: 向量维度
        :param hidden_dim: 隐藏层维度
        '''
        super(GCN4Rec, self).__init__()

        # 随机初始化所有用户向量
        self.users = nn.Embedding(n_users, dim, max_norm = 1)
        # 随机初始化所有节点向量，其中包含了实体的向量
        self.entitys = nn.Embedding(n_entitys, dim, max_norm = 1)

        # 记录下所有节点索引
        self.all_entitys_indexes = torch.LongTensor(range(n_entitys))

        # 初始化两个 GCN 层
        self.conv1 = SAGEConv(dim, hidden_dim, 'mean')  # 128 -> 64
        self.conv2 = SAGEConv(hidden_dim, dim, 'mean')  # 64 -> 128

    def gnnForward(self, i, g):
        '''
        :param i: 物品索引 [ batch_size, ]
        :param edges: 表示图的边集
        '''
        # [ n_entitys, dim ]  item_embedding 矩阵
        """
        每次读入 item_embedding 矩阵，当物品特别多时，内存加载不了包含所有物品及所有物品特征的节点向量，
        而优化办法是每次仅传入会参与计算的节点向量，但是索引方面需要书写专门的处理逻辑。
        """
        x = self.entitys(self.all_entitys_indexes)
        # 所有节点向量进行 GCN 传播，用表示采样后的子图边集，也就是 edges 来控制小批量的计算
        # x 是所有实体点的特征，edges 是经过 graphsage 采样得到的子图，conv1 只会训练给定的子图
        x = F.dropout(F.relu(self.conv1(g, x)))    # [ n_entitys, hidden_dim ]
        x = self.conv2(g, x)      # [ n_entitys, dim ]
        # 通过物品的索引取出 [ batch_size, dim ] 形状的张量表示该批次的物品
        return x[i]

    def forward(self, u, i, g):
        # [batch_size, dim]
        items = self.gnnForward(i, g)
        # [batch_size, dim]
        users = self.users(u)
        # [batch_size, 1]
        uv = torch.sum(users * items, dim=1)
        # [batch_size, 1]
        logit = torch.sigmoid(uv)
        return logit
