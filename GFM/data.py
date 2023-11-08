import random

import dgl
import torch
import numpy as np
from tqdm import tqdm     # 产生进度条的库
import networkx as nx
import pandas as pd


def readTriple(path, sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines=line.strip().split()
            if len(lines)!=3:continue
            yield lines     # 生成器返回

def readRecData(path = "./data/rating_index.tsv", test_ratio=0.1):
    print( '读取用户评分三元组...' )
    user_set, item_set = set(), set()
    triples = []
    for u, i, r in tqdm(readTriple(path)):
        user_set.add(int(u))
        item_set.add(int(i))
        triples.append((int(u), int(i), int(r)))

    test_set = random.sample(triples, int(len(triples) * test_ratio))   # test data
    train_set = list(set(triples) - set(test_set))                        # train data
    # 返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set), list(item_set), train_set, test_set

def readGraphData(path="./data/kg_index.tsv"):
    print('读取图数据...')
    entity_set = set()     # node : item + item_feature
    pairs = []
    src = []
    tar = []
    for h, _, t in tqdm(readTriple(path)):  # node_item, relation, node_item_feature
        entity_set.add(int(h))
        entity_set.add(int(t))
        pairs.append((int(h), int(t)))
        src.append(int(h))
        tar.append(int(t))
    return list(entity_set), list(set(pairs)), src, tar

# 根据边集生成 dgl 的图
def get_graph(src, tar):
    u, v = torch.tensor(src), torch.tensor(tar)
    g = dgl.graph((u, v))
    g = dgl.to_bidirected(g)  # 创建无向图

    # len1 = len(set(set(src) | set(tar)))
    # print(g.num_nodes() == len1)    # True
    # print(g.nodes())                # [0, ..., 65609]

    return g

# 传入图与物品索引得到 torch 形式的边集
# def graphsage(g, items, n_size=5, n_deep=2):  # graphsage 在 PyG 也有现成的 API，这里自己实现，API 反而没那么灵活
#     '''
#     :param G: dgl 的图结构数据
#     :param items: 每一批次得到的物品索引
#     :param n_size: 每次采样的邻居数
#     :param n_deep: 采样的深度或者说阶数
#     :return: torch.tensor 类型的边集
#     '''
#     leftEdges = []
#     rightEdges = []
#
#     for _ in range(n_deep):
#         # 初始的节点指定为传入的物品，之后每次的初始节点为前一次采集到的邻居节点
#         target_nodes = list(set(items))
#         items = set()
#         for i in target_nodes:
#             # neighbors = list(g.neighbors(i))
#             neighbors = g.sample_neighbors([i], -1)
#             neighbors = neighbors.edges()[0].numpy()
#             if len(neighbors) >= n_size: # 如果邻居数大于指定个数则仅采样指定个数的邻居节点
#                 neighbors = np.random.choice(neighbors, size=n_size, replace=False)
#             rightEdges.extend(neighbors)                  # tar
#             leftEdges.extend([ i for _ in neighbors ])    # src
#             # 将邻居节点存下以便采样下一阶邻居时提取
#             items |= set(neighbors)
#     edges = torch.tensor([leftEdges, rightEdges], dtype=torch.long )   # 子图边集
#     return edges
