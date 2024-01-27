import random
import torch
from torch.distributions import Categorical
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
                rel_id = GLOBAL.ent2rel[str(tail.detach().item())]
            else:
                new_head = head
                new_tail = random.randint(0, num_entity - 1)
                rel_id = GLOBAL.ent2rel[str(head.detach().item())]
            anomaly = torch.tensor([new_head, new_tail], device=GLOBAL.DEVICE)
            rel_temp = GLOBAL.edges[torch.tensor(rel_id, device=GLOBAL.DEVICE)]

            if neg_triples is None:
                if (torch.sum(torch.eq(rel_temp, anomaly), dim=1) != 2).all():
                    break
            else:
                if (torch.sum(torch.eq(rel_temp, anomaly), dim=1) != 2).all() and (
                    torch.sum(torch.eq(neg_triples, anomaly), dim=1) != 2).all():
                    break
        if neg_triples is None:
            neg_triples = anomaly.unsqueeze(dim=0)
        else:
            neg_triples = torch.cat([neg_triples, anomaly.unsqueeze(dim=0)], dim=0)
    return neg_triples


def getNeighborId(entity_id: torch.Tensor, neg_triples: torch.Tensor) -> torch.Tensor:
    """
    获取序号为entity_id的实体在正负三元组数据集中的邻居的序号
    """
    n = torch.tensor(GLOBAL.ent2rel[str(entity_id.detach().item())], device=GLOBAL.DEVICE)
    a = torch.nonzero(neg_triples[:, 0] == entity_id).squeeze(dim=1)
    b = torch.nonzero(neg_triples[:, 1] == entity_id).squeeze(dim=1)
    return torch.cat([n, a, b])


def getTripleNeighbor(edge_id: torch.Tensor, edges_with_anomaly: torch.Tensor, neg_triples: torch.Tensor) -> torch.Tensor:
    """
    边序号为edge_id的三元组分别在两个视图、包括edges和neg_triples里的num_neighbor个邻居

    Returns:
        组成batch的三元组的id: (2, num_neighbor+1)
    """
    num_neighbor = GLOBAL.args.num_neighbor

    head_neighbor_ids = getNeighborId(edges_with_anomaly[edge_id][0], neg_triples)
    tail_neighbor_ids = getNeighborId(edges_with_anomaly[edge_id][1], neg_triples)
    head_neighbor_ids[head_neighbor_ids != edge_id]
    tail_neighbor_ids[tail_neighbor_ids != edge_id]
    head_neighbor_ids.device

    if head_neighbor_ids.shape[0] >= num_neighbor:
        head_neighbor_ids = torch.randperm(head_neighbor_ids.shape[0], device=GLOBAL.DEVICE)[:num_neighbor]
    elif head_neighbor_ids.shape[0] > 0:
        m = Categorical(torch.ones(head_neighbor_ids.shape[0]))
        head_neighbor_ids = head_neighbor_ids[m.sample((num_neighbor,))]
    else:
        head_neighbor_ids = torch.tensor([edge_id], device=GLOBAL.DEVICE).repeat(num_neighbor, 1).squeeze(dim=1)

    if tail_neighbor_ids.shape[0] >= num_neighbor:
        tail_neighbor_ids = torch.randperm(tail_neighbor_ids.shape[0], device=GLOBAL.DEVICE)[:num_neighbor]
    elif tail_neighbor_ids.shape[0] > 0:
        m = Categorical(torch.ones(tail_neighbor_ids.shape[0]))
        tail_neighbor_ids = tail_neighbor_ids[m.sample((num_neighbor,))]
    else:
        tail_neighbor_ids = torch.tensor([edge_id], device=GLOBAL.DEVICE).repeat(num_neighbor, 1).squeeze(dim=1)

    head_neighbor_ids = torch.cat([torch.tensor([edge_id], device=GLOBAL.DEVICE), head_neighbor_ids])
    tail_neighbor_ids = torch.cat([torch.tensor([edge_id], device=GLOBAL.DEVICE), tail_neighbor_ids])

    return torch.stack([head_neighbor_ids, tail_neighbor_ids])


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
    neg_triples = generateAnomalousTriples(ids)
    edges_with_anomaly = torch.cat([GLOBAL.edges, neg_triples], dim=0)

    ids += list(range(rel_num, rel_num + len(ids)))

    # (batch_size*2, 2, num_neighbor+1)
    batch_triples_id = torch.stack(list(map(lambda x: getTripleNeighbor(x, edges_with_anomaly, neg_triples), ids)))

    batch_h = GLOBAL.node_embed[edges_with_anomaly[:, 0][batch_triples_id]]
    batch_t = GLOBAL.node_embed[edges_with_anomaly[:, 1][batch_triples_id]]

    # 异常三元组的边权重为其在edges_with_anomaly的索引减去edges的长度
    batch_triples_id = batch_triples_id * (batch_triples_id < rel_num) + torch.max((batch_triples_id * (batch_triples_id >= rel_num) - rel_num), torch.tensor([0], device=GLOBAL.DEVICE))
    batch_r = GLOBAL.edge_weights[batch_triples_id]

    return batch_h, batch_r, batch_t, len(ids)
