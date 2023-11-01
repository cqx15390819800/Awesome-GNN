import math
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, g, in_features_dim, out_features_dim, activation=F.relu, bias=True):
        super(GraphConvolution, self).__init__()

        self.g = g

        self.in_features = in_features_dim
        self.out_features = out_features_dim

        self.weight = nn.Linear(in_features_dim, out_features_dim, bias=bias)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight.weight)  # 保持前向传播和反向传播中的梯度方差大致相等，从而避免梯度爆炸或梯度消失的问题
        if self.weight.bias is not None:
            init.constant_(self.weight.bias, 0)

    def edge_attention(self, edges):
        src_out_degree = edges.src['out_degree_norm']
        dst_out_degree = edges.dst['out_degree_norm']
        w = src_out_degree * dst_out_degree
        return {'w': w}     # (13264, 1)

    def message_func(self, edges):
        return {'feat': edges.src['feat'], 'w': edges.data['w']}      # feat、w 作为 message

    def reduce_func(self, nodes):
        w = nodes.mailbox['w']        # 边权  (num_node, 入边数量, 1)
        # nodes.mailbox['feat'] = (num_node, 入边数量, emb_dim)
        h = torch.sum(w * nodes.mailbox['feat'], dim=1)    # 加权求和
        return {'feat': h}     # 返回当前点的 embedding = (1, out_feats)

    def forward(self, h):
        # h = (num_node, in_features_dim)
        h = self.weight(h)
        self.g.ndata['feat'] = h
        self.g.apply_edges(self.edge_attention)     # 边权（度数归一化）
        self.g.update_all(self.message_func, self.reduce_func)  # message-passing  a = D^(-1/2) * A * D^(-1/2) * H(l)

        if self.activation is not None:
            self.g.ndata['feat'] = self.activation(self.g.ndata['feat'])

        return self.g.ndata['feat']    # (num_node, out_features_dim)

class GCN(nn.Module):
    def __init__(self, g, in_features_dim, hid_features_dim, out_features_dim, n_layers, dropout=0.5):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphConvolution(g, in_features_dim, hid_features_dim, F.relu, bias=False))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConvolution(g, hid_features_dim, hid_features_dim, F.relu, bias=False))

        # output layer
        self.layers.append(GraphConvolution(g, hid_features_dim, out_features_dim, bias=False))     # 最后一层不能过激活函数
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, feature):
        h = feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
        return h
