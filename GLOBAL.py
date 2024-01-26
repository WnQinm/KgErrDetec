import torch

# edges: (edge_num, 2)
# edge_weights: (edge_num, 15)
# node_embed(PAUSE): (node_num, 32)
edges = torch.load("./edges.pt", map_location=torch.device("cpu"))
edge_weights = torch.load("./edges_weight.pt", map_location=torch.device('cpu')).to_dense()
node_embed = torch.load("./nodes_feature_pause.pt", map_location=torch.device('cpu'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')