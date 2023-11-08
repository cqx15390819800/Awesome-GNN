import torch
from torch import nn

class GFM(nn.Module):

    def __init__(self, n_users, n_entitys, dim, hidden_dim):
        '''
        :param n_users: 用户数量
        :param n_entitys: 实体数量(物品+物品特征)
        :param dim: 向量维度
        :param hidden_dim: 隐藏层维度
        '''
        super(GFM, self).__init__()

        # 随机初始化所有用户向量
        self.users = nn.Embedding(n_users, dim, max_norm = 1)
        # 随机初始化所有节点向量，其中包含了实体的向量
        self.entitys = nn.Embedding(n_entitys, dim, max_norm = 1)

        # 记录下所有节点索引
        self.all_entitys_indexes = torch.LongTensor(range(n_entitys))

    def FMaggregator(self, target_embs, neighbor_entitys_embeddings):
        '''
        :param target_embeddings: 目标节点的向量 [ batch_size, dim ]
        :param neighbor_entitys_embeddings: 目标节点的邻居节点向量 [ batch_size, n_neighbor, dim ]
        '''
        # neighbor_entitys_embeddings:[batch_size, n_neighbor, dim]
        # [batch_size, dim]
        square_of_sum = torch.sum(neighbor_entitys_embeddings, dim=1) ** 2
        # [batch_size, dim]
        sum_of_square = torch.sum(neighbor_entitys_embeddings ** 2, dim=1)
        # [batch_size, dim]
        output = square_of_sum - sum_of_square
        # return torch.add(output, target_embs)
        return output + target_embs     # FM + self-embedding

    def message_func(self, edges):
        return {'h': edges.src['feat']}

    def reduce_func(self, nodes):
        # FM_aggregator
        neighbor_entitys_embeddings = nodes.mailbox['h']
        target_embs = nodes.data['feat']
        h = self.FMaggregator(target_embs, neighbor_entitys_embeddings)   # FM + self-embedding
        return {'feat': h}     # 返回当前点的 embedding = (1, out_feats)

    def gnnforward(self, i, blocks):
        '''
        :param i: 物品索引 [ batch_size, ]
        :param edges: 表示图的边集
        '''
        # [ n_entitys, dim ]  item_embedding 矩阵
        """
        每次读入 item_embedding 矩阵，当物品特别多时，内存加载不了包含所有物品及所有物品特征的节点向量，
        而优化办法是每次仅传入会参与计算的节点向量，但是索引方面需要书写专门的处理逻辑。
        """

        x = self.entitys(self.all_entitys_indexes)  # all_embedding

        for k in range(blocks.__len__()):     # 0 1
            h_src = x[blocks[k].srcdata["_ID"]]
            h_dst = x[blocks[k].dstdata["_ID"]]
            blocks[k].srcdata['feat'] = h_src
            blocks[k].dstdata['feat'] = h_dst
            blocks[k].update_all(self.message_func, self.reduce_func)
            # blocks[k].update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))
            x[blocks[k].srcdata["_ID"]] = blocks[k].srcdata['feat']   # update

        return x[i]

    def forward(self, u, i, block):
        # with block.local_scope():  # 不改变原图的写法（子图常用）
        # [batch_size, dim]
        items = self.gnnforward(i, block)
        # [batch_size, dim]
        users = self.users(u)
        # [batch_size, 1]
        uv = torch.sum(users * items, dim=1)
        # [batch_size, 1]
        logit = torch.sigmoid(uv)
        return logit
