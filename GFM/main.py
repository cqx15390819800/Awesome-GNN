import torch

from utils import train
from data import readRecData, readGraphData, get_graph

def main():
    # 返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表    rating_index.tsv
    user_set, item_set, train_set, test_set = readRecData()
    # 读取所有节点索引及表示物品全量图的边集对    kg_index.tsv
    entitys, pairs, src, tar = readGraphData()
    # 传入边集得到 dgl 的图结构数据
    g = get_graph(src, tar)

    train(g, user_set, train_set, test_set, entitys)


if __name__ == '__main__':
    main()

"""
epoch 4, loss = 0.5787
train: Precision 0.7578 | Recall 0.8449 | accuracy 0.7662
test: Precision 0.6850 | Recall 0.7943 | accuracy 0.6962
"""
