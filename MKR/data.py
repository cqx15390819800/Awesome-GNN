import torch
from torch.utils.data import Dataset, DataLoader
import sys
import random
import copy

def readTriple(path, sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines=line.strip().split()
            if len(lines) != 3: continue
            yield lines     # 生成器返回

def readKGData(path="./data/kg_index.tsv"):
    print('读取知识图谱数据...')
    entity_set = set()
    relation_set = set()
    triples = []
    for h, r, t in readTriple(path):    # h, r, t 表示的是电影之间的关系，都是 item
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])
    return list(entity_set), list(relation_set), triples

def readRecData(path="./data/rating_index.tsv", test_ratio=0.1):
    print('读取用户评分三元组...')
    user_set, item_set = set(), set()
    triples = []
    for u, i, r in readTriple(path):
        user_set.add(int(u))
        item_set.add(int(i))
        triples.append((int(u), int(i), int(r)))

    test_set = random.sample(triples, int(len(triples) * test_ratio))
    train_set = list(set(triples) - set(test_set))
    #返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set), list(item_set), train_set, test_set

def graph(path="./data/kg_index.tsv"):
    print('读取知识图谱数据...')
    data_dict = {}  # key: [[h, ...], [t, ...]]
    for h, r, t in readTriple(path):    # h, r, t 表示的是电影之间的关系，都是 item
        if r not in data_dict:
            data_dict[r] = [[], []]
        data_dict[r][0].append(torch.tensor(int(h)))
        data_dict[r][1].append(torch.tensor(int(t)))
    return data_dict

# 继承 torch 自带的 Dataset 类, 重构 __getitem__ 与 __len__ 方法
class KgDatasetWithNegativeSampling(Dataset):

    def __init__(self, triples, entitys):
        self.triples = triples  # 知识图谱 HRT 三元组
        self.entitys = entitys  # 所有实体集合列表

    def __getitem__(self, index):
        '''
        :param index: 一批次采样的列表索引序号
        '''
        # 根据索引取出正例
        pos_triple = self.triples[index]
        # 通过负例采样的方法得到负例
        neg_triple = self.negtiveSampling(pos_triple)
        return pos_triple, neg_triple

    # 负例采样方法
    def negtiveSampling(self, triple):
        seed = random.random()
        neg_triple = copy.deepcopy(triple)
        if seed > 0.5:  # 替换 head
            rand_head = triple[0]
            while rand_head == triple[0]:  # 如果采样得到自己则继续循环
                # 从所有实体中随机采样一个实体
                rand_head = random.sample(self.entitys, 1)[0]
            neg_triple[0] = rand_head
        else:   # 替换 tail
            rand_tail = triple[2]
            while rand_tail == triple[2]:
                rand_tail = random.sample(self.entitys, 1)[0]
            neg_triple[2] = rand_tail
        return neg_triple

    def __len__(self):
        return len(self.triples)


if __name__ == '__main__':
    # 读取文件得到所有实体列表，所有关系列表，以及 HRT 三元组列表
    entitys, relations, triples = readKGData()
    # 传入 HRT 三元与所有实体得到包含正例与负例三元组的 data set
    train_set = KgDatasetWithNegativeSampling(triples, entitys)
    # 通过 torch 的 DataLoader 方法批次迭代三元组数据
    for set in DataLoader(train_set, batch_size = 8, shuffle=True):
        # 将正负例数据拆解开
        pos_set, neg_set = set
        # 可以打印一下看看
        print(pos_set)
        print(neg_set)
        sys.exit()
