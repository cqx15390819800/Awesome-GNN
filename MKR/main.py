from utils import train
import dgl

from data import readKGData, KgDatasetWithNegativeSampling, readRecData


def main():

    # 读取所有节点索引及表示物品全量图的边集对    kg_index.tsv
    entitys, relation, triples = readKGData()
    kgTrainSet = KgDatasetWithNegativeSampling(triples, entitys)
    users, items, train_set, test_set = readRecData()
    train(entitys, relation, triples, kgTrainSet, users, items, train_set, test_set)


if __name__ == '__main__':
    main()

"""
epoch 1,avg_loss=0.0120
train: Precision 0.5634 | Recall 0.9847 | accuracy 0.5658
test: Precision 0.5547 | Recall 0.9845 | accuracy 0.5583
"""
