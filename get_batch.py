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
    Args:
        batch_id: 第几个batch
        seq: 三元组索引序列, 用于按照顺序随机取batch size个数据, 每个epoch开始会打乱一次

    Returns:
        batch_h: (batch_size, 2, num_neighbor+1, node_embed_dim)
        batch_r: (batch_size, 2, num_neighbor+1, 15)
        batch_t: (batch_size, 2, num_neighbor+1, node_embed_dim)

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

    # (batch_size, 2, num_neighbor+1)
    batch_triples_id = list(map(lambda x: getTripleNeighbor(x, num_neighbor), ids))
    batch_triples_id = torch.tensor(batch_triples_id)

    batch_h = GLOBAL.node_embed[GLOBAL.edges[:, 0][batch_triples_id]]
    batch_r = GLOBAL.edge_weights[batch_triples_id]
    batch_t = GLOBAL.node_embed[GLOBAL.edges[:, 1][batch_triples_id]]

    return batch_h, batch_r, batch_t, len(ids)
