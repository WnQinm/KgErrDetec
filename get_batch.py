import random
import torch
import GLOBAL
from typing import List, Tuple


def getNeighborId(entity_id: int) -> List[int]:
    a = torch.nonzero(GLOBAL.edges[:, 0] == entity_id).squeeze().tolist()
    b = torch.nonzero(GLOBAL.edges[:, 1] == entity_id).squeeze().tolist()
    if type(a) is int:
        a = [a]
    if type(b) is int:
        b = [b]
    return a + b


def getTripleNeighbor(edge_id: int, num_neighbor: int) -> List[List[int]]:
    """
    边序号为edge_id的三元组分别在两个视图的num_neighbor个邻居

    Returns:
        组成batch的三元组的id: (2, num_neighbor+1)
    """
    head_neighbor_ids = getNeighborId(GLOBAL.edges[edge_id][0])
    tail_neighbor_ids = getNeighborId(GLOBAL.edges[edge_id][1])
    head_neighbor_ids.remove(edge_id)
    tail_neighbor_ids.remove(edge_id)

    if len(head_neighbor_ids) >= num_neighbor:
        head_neighbor_ids = random.sample(head_neighbor_ids, k=num_neighbor)
    elif len(head_neighbor_ids) > 0:
        head_neighbor_ids = random.choices(head_neighbor_ids, k=num_neighbor)
    else:
        head_neighbor_ids = [edge_id] * num_neighbor

    if len(tail_neighbor_ids) >= num_neighbor:
        tail_neighbor_ids = random.sample(tail_neighbor_ids, k=num_neighbor)
    elif len(tail_neighbor_ids) > 0:
        tail_neighbor_ids = random.choices(tail_neighbor_ids, k=num_neighbor)
    else:
        tail_neighbor_ids = [edge_id] * num_neighbor

    return [[edge_id] + head_neighbor_ids, [edge_id] + tail_neighbor_ids]


def get_pair_batch(
    batch_id:int, seq:List[int], batch_size:int, num_neighbor:int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    论文代码的损失函数有点抽象, 和论文中写的好像不太一样, 以下是我的猜测:

    使用了2*batch_size个数据的损失更新GAT, (x1,z1),(x2,z2)\\
    batch_size个数据的损失更新LSTM, (hrt, h'r't')

    损失函数 max(0, pos_loss - neg_loss + margin)\\
    其中pos_loss = sim(x1,z1) + E(hrt), neg_loss = sim(x2,z2) + E(h'r't')\\
    由于sim(x1,z1)和sim(x2,z2)的目的三元组在两个视图的距离尽可能小, 它们最优情况为0\\
    此时Loss即为公式(8)的MarginRankingLoss

    但是在论文代码中似乎并没有看到如论文3.3.1所述去生成一个错误的三元组h'r't', 
    而是直接使用了真实的h'r't', 显然再加margin就不太合理了, 
    不过这里的实现还是按照论文代码get_pair_batch_train_common中"可能错误"的方式

    Args:
        batch_id: 第几个batch
        seq: list(range(GLOBAL.edges.shape[0] // 2))
        三元组索引序列, 用于按照顺序随机取batch size个数据, 每个epoch开始会打乱一次

    Returns:
        batch_h: (batch_size*2, 2, num_neighbor+1, node_embed_dim)
        batch_r: (batch_size*2, 2, num_neighbor+1, 15)
        batch_t: (batch_size*2, 2, num_neighbor+1, node_embed_dim)

        其中node_embed_dim为CompanyKG数据集实体嵌入维度,
        选用不同的模型有不同的维度, 例如PAUSE为32维;
    """
    # 每个epoch开头再重新打乱一下
    if batch_id == 0:
        random.shuffle(seq)

    # 取batch_size个数据
    if (batch_id + 1) * batch_size > len(seq):
        ids = seq[batch_id * batch_size :]
    else:
        ids = seq[batch_id * batch_size : (batch_id + 1) * batch_size]

    # 取2*batch_size个数据
    ids += [i + len(seq) for i in ids]

    # (batch_size*2, 2, num_neighbor+1)
    batch_triples_id = list(map(lambda x: getTripleNeighbor(x, num_neighbor), ids))
    batch_triples_id = torch.tensor(batch_triples_id)

    batch_h = GLOBAL.node_embed[GLOBAL.edges[:, 0][batch_triples_id]]
    batch_r = GLOBAL.edge_weights[batch_triples_id]
    batch_t = GLOBAL.node_embed[GLOBAL.edges[:, 1][batch_triples_id]]

    return batch_h, batch_r, batch_t, len(ids)
