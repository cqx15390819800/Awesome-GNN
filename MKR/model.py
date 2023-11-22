import torch
from torch import nn
from torch.nn import Parameter, init


class CrossCompress(nn.Module):
    def __init__(self, dim):
        super(CrossCompress, self).__init__()
        self.dim = dim

        self.weight_vv = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))
        self.weight_ev = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))
        self.weight_ve = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))
        self.weight_ee = init.xavier_uniform_(Parameter(torch.empty(dim, 1)))

        self.bias_v = init.xavier_uniform_(Parameter(torch.empty(1, dim)))
        self.bias_e = init.xavier_uniform_(Parameter(torch.empty(1, dim)))

    def forward( self, v, e ):
        v = v.reshape(-1, self.dim, 1)    # [batch_size, dim, 1]
        e = e.reshape(-1, 1, self.dim)    # [batch_size, 1, dim]
        c_matrix = torch.matmul(v, e)     # [batch_size, dim, dim]
        c_matrix_transpose = torch.transpose(c_matrix, dim0=1, dim1=2)    # [batch_size, dim, dim]
        c_matrix = c_matrix.reshape((-1, self.dim))     # [batch_size * dim, dim]
        c_matrix_transpose = c_matrix_transpose.reshape((-1, self.dim))  # [batch_size * dim, dim]
        # [batch_size, dim]
        v_output = torch.matmul(c_matrix, self.weight_vv) + torch.matmul(c_matrix_transpose, self.weight_ev)
        e_output = torch.matmul(c_matrix, self.weight_ve) + torch.matmul(c_matrix_transpose, self.weight_ee)
        # [batch_size, dim]
        v_output = v_output.reshape(-1, self.dim) + self.bias_v
        e_output = e_output.reshape(-1, self.dim) + self.bias_e
        return v_output, e_output

# 附加 Dropout 的全连接网络层
class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob):
        super(DenseLayer, self).__init__()
        self.liner = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x, isTrain):
        out = torch.relu(self.liner(x))
        if isTrain:     # 训练时加入 dropout 防止过拟合
            out = self.drop(out)
        return out

class MKR(nn.Module):
    def __init__(self, n_users, n_entitys, n_relations, dim=128, margin=1, alpha=0.2, dropout_prob=0.5):
        super().__init__()
        self.margin = margin
        self.u_emb = nn.Embedding(n_users, dim)       # 用户向量
        self.e_emb = nn.Embedding(n_entitys, dim)     # 实体向量
        self.r_emb = nn.Embedding(n_relations, dim)   # 关系向量

        self.user_dense1 = DenseLayer(dim, dim, dropout_prob)
        self.user_dense2 = DenseLayer(dim, dim, dropout_prob)
        self.user_dense3 = DenseLayer(dim, dim, dropout_prob)
        self.tail_dense1 = DenseLayer(dim, dim, dropout_prob)
        self.tail_dense2 = DenseLayer(dim, dim, dropout_prob)
        self.tail_dense3 = DenseLayer(dim, dim, dropout_prob)
        self.cc_unit1 = CrossCompress(dim)
        self.cc_unit2 = CrossCompress(dim)
        self.cc_unit3 = CrossCompress(dim)

        self.BCEloss = nn.BCELoss()

        self.alpha = alpha    # kge 损失函数的计算权重

    def hinge_loss(self, y_pos, y_neg):
        dis = y_pos - y_neg + self.margin
        return torch.sum(torch.relu(dis))

    # kge 采用最基础的 TransE 算法
    def TransE(self, h, r, t):
        score = h + r - t
        return torch.sum(score**2, dim=1)**0.5

    # 前向传播
    def forward(self, rec_set, kg_set, isTrain=True):
        # 推荐预测部分的提取初始 embedding
        u, v ,y = rec_set
        y = torch.FloatTensor(y.detach().numpy())
        u = self.u_emb(u)
        v = self.e_emb(v)

        # 分开知识图谱三元组的正负例
        x_pos, x_neg = kg_set

        # 提取知识图谱三元组正例 h,r,t 的初始 embedding
        h_pos, r_pos, t_pos = x_pos
        h_pos = self.e_emb(h_pos)
        r_pos = self.r_emb(r_pos)
        t_pos = self.e_emb(t_pos)

        # 提取知识图谱三元组负例 h,r,t 的初始 embedding
        h_neg, r_neg, t_neg = x_neg
        h_neg = self.e_emb(h_neg)
        r_neg = self.r_emb(r_neg)
        t_neg = self.e_emb(t_neg)

        # 将用户向量经三层全连接层传递
        u = self.user_dense1(u, isTrain)
        u = self.user_dense2(u, isTrain)
        u = self.user_dense3(u, isTrain)
        # 将 KG 正例的尾实体向量经三层全连接层传递
        t_pos = self.tail_dense1(t_pos, isTrain)
        t_pos = self.tail_dense2(t_pos, isTrain)
        t_pos = self.tail_dense3(t_pos, isTrain)

        # 将物品与 KG 正例头实体一同经三层 C 单元传递
        v, h_pos = self.cc_unit1(v, h_pos)
        v, h_pos = self.cc_unit2(v, h_pos)
        v, h_pos = self.cc_unit3(v, h_pos)

        # 计算推荐预测的预测值及损失函数
        rec_pred = torch.sigmoid(torch.sum(u*v, dim=1))
        rec_loss = self.BCEloss(rec_pred, y)

        # 计算 kg 正例的 TransE 评分
        kg_pos = self.TransE(h_pos, r_pos, t_pos)
        # 计算 kg 负例的 TransE 评分，注意负例的实体不要与物品向量一同去走 C 单元
        kg_neg = self.TransE(h_neg, r_neg, t_neg)
        # 计算 kge 的 hing loss
        kge_loss = self.hinge_loss(kg_pos, kg_neg)

        # 将推荐产生的损失函数与 kge 产生的损失函数加权相加后返回
        return rec_loss + self.alpha * kge_loss

    # 测试时用
    def predict(self, u, v, isTrain=False):
        u = self.u_emb(u)
        v = self.e_emb(v)
        u = self.user_dense1(u, isTrain)
        u = self.user_dense2(u, isTrain)
        u = self.user_dense3(u, isTrain)
        # 第一层输入 C 单元的 KG 头实体是物品自身
        v, h = self.cc_unit1(v, v)
        v, h = self.cc_unit2(v, h)
        v, h = self.cc_unit3(v, h)
        return torch.sigmoid(torch.sum(u*v, dim=1))
