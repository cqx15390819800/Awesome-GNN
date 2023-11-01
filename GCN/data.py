from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


def get_data(args):
    transform = (
        AddSelfLoop()   # 添加自环
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        return CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        return CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        return PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

