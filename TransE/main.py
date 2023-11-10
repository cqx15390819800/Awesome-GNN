from utils import train
import dgl

from data import readKGData, KgDatasetWithNegativeSampling, graph


def main():

    # 读取所有节点索引及表示物品全量图的边集对    kg_index.tsv
    entitys, relation, triples = readKGData()
    train_set = KgDatasetWithNegativeSampling(triples, entitys)
    train(entitys, relation, triples, train_set)

    # data_dict = graph()
    #
    # dict = {
    #     ('movie', key, 'movie'): (value[0], value[1])
    #     for key, value in data_dict.items()
    # }
    #
    # g = dgl.heterograph(dict)   # 异构图
    #
    # train1(g, entitys, relation, triples, train_set)

if __name__ == '__main__':
    main()
"""
epoch 4,avg_loss=0.2020
"""
