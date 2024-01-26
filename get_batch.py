import random
import torch
import GLOBAL
from typing import List, Tuple


def generateAnomalousTriples(ids: List[int]) -> torch.Tensor:
    """
    根据edges[ids]中每个三元组, 通过替换头或尾生成异常三元组,
    输出负三元组训练batch [[head, tail], ...]

    由于数据集三元组过多, 如果一次性全部生成, 用4090力所能及优化代码还是得五十多小时才能生成完毕,
    只能动态生成, 一个batch_size用4090大约2秒
    """
    neg_triples = None
    num_entity = GLOBAL.node_embed.shape[0]
    for i in ids:
        head, tail = GLOBAL.edges[i][0], GLOBAL.edges[i][1]
        head_or_tail = random.randint(0, 1)
        while True:
            if head_or_tail == 0:
                new_head = random.randint(0, num_entity - 1)
                new_tail = tail
            else:
                new_head = head
                new_tail = random.randint(0, num_entity - 1)
            anomaly = torch.tensor([new_head, new_tail]).to(GLOBAL.DEVICE)

            # 这里很慢
            if neg_triples is None:
                if (torch.sum(torch.eq(GLOBAL.edges, anomaly), dim=1) != 2).all():
                    break
            else:
                if (torch.sum(torch.eq(GLOBAL.edges, anomaly), dim=1) != 2).all() and (
                    torch.sum(torch.eq(neg_triples, anomaly), dim=1) != 2).all():
                    break
        if neg_triples is None:
            neg_triples = anomaly.unsqueeze(dim=0)
        else:
            neg_triples = torch.cat([neg_triples, anomaly.unsqueeze(dim=0)], dim=0)
    return neg_triples


def getNeighborId(entity_id: int, dataset: torch.Tensor) -> List[int]:
    """
    获取序号为entity_id的实体在正负三元组数据集中的邻居的序号
    """
    a = torch.nonzero(dataset[:, 0] == entity_id).squeeze().tolist()
    b = torch.nonzero(dataset[:, 1] == entity_id).squeeze().tolist()

    a = [a] if type(a) is int else a
    b = [b] if type(b) is int else b

    return a + b


def getTripleNeighbor(edge_id: int, dataset: torch.Tensor) -> List[List[int]]:
    """
    边序号为edge_id的三元组分别在两个视图、包括edges和neg_triples里的num_neighbor个邻居

    Returns:
        组成batch的三元组的id: (2, num_neighbor+1)
    """
    num_neighbor = GLOBAL.args.num_neighbor

    head_neighbor_ids = getNeighborId(dataset[edge_id][0], dataset)
    tail_neighbor_ids = getNeighborId(dataset[edge_id][1], dataset)
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
    batch_id: int, seq: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
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
    batch_size = GLOBAL.args.batch_size
    rel_num = GLOBAL.edges.shape[0]

    # 每个epoch开头再重新打乱一下
    if batch_id == 0:
        random.shuffle(seq)

    # 取batch_size个数据
    if (batch_id + 1) * batch_size > len(seq):
        ids = seq[batch_id * batch_size :]
    else:
        ids = seq[batch_id * batch_size : (batch_id + 1) * batch_size]

    # 生成batch_size个数据作为异常数据加入到edges最后
    edges_with_anomaly = torch.cat([GLOBAL.edges, generateAnomalousTriples(ids)], dim=0)

    ids += list(range(rel_num, rel_num + len(ids)))

    # (batch_size*2, 2, num_neighbor+1)
    batch_triples_id = list(map(lambda x: getTripleNeighbor(x, edges_with_anomaly), ids))
    batch_triples_id = torch.tensor(batch_triples_id).to(GLOBAL.DEVICE)

    batch_h = GLOBAL.node_embed[edges_with_anomaly[:, 0][batch_triples_id]]
    batch_t = GLOBAL.node_embed[edges_with_anomaly[:, 1][batch_triples_id]]

    # 异常三元组的边权重为其在edges_with_anomaly的索引减去edges的长度
    batch_triples_id = batch_triples_id * (batch_triples_id < rel_num) + torch.max((batch_triples_id * (batch_triples_id >= rel_num) - rel_num), torch.tensor([0]).to(GLOBAL.DEVICE))
    batch_r = GLOBAL.edge_weights[batch_triples_id]

    return batch_h, batch_r, batch_t, len(ids)
