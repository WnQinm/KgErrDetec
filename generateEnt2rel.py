import torch
from tqdm import tqdm
import json

edges = torch.load("./data/edges.pt", map_location=torch.device("cpu"))
node_embed = torch.load("./data/nodes_feature_pause.pt", map_location=torch.device('cpu'))

map_dic = {}
for i in tqdm(range(node_embed.shape[0])):
    map_dic[i] = []
for i in tqdm(range(edges.shape[0])):
    head = edges[i][0].detach().item()
    tail = edges[i][1].detach().item()
    map_dic[head].append(i)
    map_dic[tail].append(i)

jsonstr = json.dumps(map_dic)
f = open('ent2rel.json','w')
f.write(jsonstr)
f.close()

# 索引为字符串，须将entity_id数字转字符串

# 加载
# f = open('ent2rel.json', 'r')
# d = json.load(f)
# f.close()