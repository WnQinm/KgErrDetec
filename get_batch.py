import random
import torch
from torch.distributions import Categorical
from torch.utils.data import Dataset
import GLOBAL
from typing import List, Optional


class CompanyKgDataset(Dataset):
    def __len__(self):
        return GLOBAL.edges.shape[0]
    
    def __getitem__(self, index):
        return index


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
            anomaly = torch.tensor([new_head, new_tail])
            rel_temp = GLOBAL.edges[torch.tensor(rel_id)]

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


def getNeighborId(entity_id: torch.Tensor, neg_triples: Optional[torch.Tensor]) -> torch.Tensor:
    """
    获取序号为entity_id的实体在正负三元组数据集中的邻居的序号
    """
    n = torch.tensor(GLOBAL.ent2rel[str(entity_id.detach().item())], dtype=torch.int64)
    if neg_triples is None:
        return n
    else:
        a = torch.nonzero(neg_triples[:, 0] == entity_id).squeeze(dim=1) + GLOBAL.edges.shape[0]
        b = torch.nonzero(neg_triples[:, 1] == entity_id).squeeze(dim=1) + GLOBAL.edges.shape[0]
        a = a.long()
        b = b.long()
        return torch.cat([n, a, b])


def getTripleNeighbor(edge_id: torch.Tensor, edges_with_anomaly: torch.Tensor, neg_triples: Optional[torch.Tensor]) -> torch.Tensor:
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

    if head_neighbor_ids.shape[0] >= num_neighbor:
        head_neighbor_ids = torch.randperm(head_neighbor_ids.shape[0])[:num_neighbor]
    elif head_neighbor_ids.shape[0] > 0:
        m = Categorical(torch.ones(head_neighbor_ids.shape[0]))
        head_neighbor_ids = head_neighbor_ids[m.sample((num_neighbor,))]
    else:
        head_neighbor_ids = torch.tensor([edge_id]).repeat(num_neighbor, 1).squeeze(dim=1)

    if tail_neighbor_ids.shape[0] >= num_neighbor:
        tail_neighbor_ids = torch.randperm(tail_neighbor_ids.shape[0])[:num_neighbor]
    elif tail_neighbor_ids.shape[0] > 0:
        m = Categorical(torch.ones(tail_neighbor_ids.shape[0]))
        tail_neighbor_ids = tail_neighbor_ids[m.sample((num_neighbor,))]
    else:
        tail_neighbor_ids = torch.tensor([edge_id]).repeat(num_neighbor, 1).squeeze(dim=1)

    head_neighbor_ids = torch.cat([torch.tensor([edge_id]), head_neighbor_ids])
    tail_neighbor_ids = torch.cat([torch.tensor([edge_id]), tail_neighbor_ids])

    return torch.stack([head_neighbor_ids, tail_neighbor_ids])


def companyKgTrainDataCollator(batch):
    ids = batch
    rel_num = GLOBAL.edges.shape[0]

    # 生成batch_size个数据作为异常数据加入到edges最后
    neg_triples = generateAnomalousTriples(ids)
    edges_with_anomaly = torch.cat([GLOBAL.edges, neg_triples], dim=0)

    ids += list(range(rel_num, rel_num + len(ids)))

    # (batch_size*2, 2, num_neighbor+1)
    batch_triples_id = torch.stack(list(map(lambda x: getTripleNeighbor(x, edges_with_anomaly, neg_triples), ids)))

    batch_h = GLOBAL.node_embed[edges_with_anomaly[:, 0][batch_triples_id]]
    batch_t = GLOBAL.node_embed[edges_with_anomaly[:, 1][batch_triples_id]]

    # 异常三元组的边权重为其在edges_with_anomaly的索引减去edges的长度
    batch_triples_id = batch_triples_id * (batch_triples_id < rel_num) + torch.max((batch_triples_id * (batch_triples_id >= rel_num) - rel_num), torch.tensor([0]))
    batch_r = GLOBAL.edge_weights[batch_triples_id]

    return batch_h, batch_r, batch_t


def companyKgTestDataCollator(batch):
    ids = batch
    rel_num = GLOBAL.edges.shape[0]

    # 生成batch_size个数据作为异常数据加入到edges最后
    neg_triples = generateAnomalousTriples(ids)
    edges_with_anomaly = torch.cat([GLOBAL.edges, neg_triples], dim=0)

    labels = torch.cat([torch.ones(len(ids), dtype=torch.int), torch.zeros(len(ids), dtype=torch.int)])

    ids += list(range(rel_num, rel_num + len(ids)))

    # (batch_size*2, 2, num_neighbor+1)
    batch_triples_id = torch.stack(list(map(lambda x: getTripleNeighbor(x, edges_with_anomaly, neg_triples), ids)))

    shuffle = torch.randperm(len(ids))
    labels = labels[shuffle]
    batch_triples_id = batch_triples_id[shuffle]

    batch_h = GLOBAL.node_embed[edges_with_anomaly[:, 0][batch_triples_id]]
    batch_t = GLOBAL.node_embed[edges_with_anomaly[:, 1][batch_triples_id]]

    # 异常三元组的边权重为其在edges_with_anomaly的索引减去edges的长度
    batch_triples_id = batch_triples_id * (batch_triples_id < rel_num) + torch.max((batch_triples_id * (batch_triples_id >= rel_num) - rel_num), torch.tensor([0]))
    batch_r = GLOBAL.edge_weights[batch_triples_id]

    return batch_h, batch_r, batch_t, labels


def companyKgFinalDataCollator(batch):
    ids = batch
    batch_triples_id = torch.stack(list(map(lambda x: getTripleNeighbor(x, GLOBAL.edges, None), ids)))
    batch_h = GLOBAL.node_embed[GLOBAL.edges[:, 0][batch_triples_id]]
    batch_r = GLOBAL.edge_weights[batch_triples_id]
    batch_t = GLOBAL.node_embed[GLOBAL.edges[:, 1][batch_triples_id]]
    # (batch_size, 2, num_neighbor+1, embed_dim)
    return batch_h, batch_r, batch_t