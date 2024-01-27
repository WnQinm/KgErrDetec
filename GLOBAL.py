import torch
import argparse
import os
import json

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--mode', default='train', choices=['train', 'test'], help='run training or evaluation')
parser.add_argument('--save_dir', default=f'./checkpoints/', help='model output directory')
parser.add_argument('--load_model_path', default=f'./checkpoints/')
parser.add_argument('--save_iter', default=100, help='How many iterations to save the checkpoint')

# data
parser.add_argument('--data_path', default=f'./data/', help='path to the dataset')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')

# model architecture
parser.add_argument('--BiLSTM_hidden_size', default=32, type=int, help='BiLSTM hidden size, 模型输出的特征维度将是hidden_size*2*3')
parser.add_argument('--BiLSTM_num_layers', default=2, type=int, help='BiLSTM layers')
parser.add_argument('--num_neighbor', default=39, type=int, help='number of neighbors')
parser.add_argument('--embedding_dim', default=32, type=int, help='实体和关系需要嵌入到同一维度')

# regularization
parser.add_argument('--alpha', type=float, default=0.2, help='hyperparameter alpha')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout for EaGNN')

# optimization
parser.add_argument('--max_epoch', default=1, help='max epochs')
parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate')
parser.add_argument('--gama', default=0.5, type=float, help="margin parameter")
parser.add_argument('--lam', default=0.1, type=float, help="trade-off parameter")
parser.add_argument('--mu', default=0.001, type=float, help="gated attention parameter")
args = parser.parse_args()

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

# edges: (edge_num, 2)
# edge_weights: (edge_num, 15)
# node_embed(PAUSE): (node_num, 32)
edges = torch.load(os.path.join(args.data_path, "edges.pt"), map_location=DEVICE).to(torch.int32)
edge_weights = torch.load(os.path.join(args.data_path, "edges_weight.pt"), map_location=DEVICE).to_dense().to(torch.float32)
node_embed = torch.load(os.path.join(args.data_path, "nodes_feature_pause.pt"), map_location=DEVICE).to(torch.float32)

f = open(os.path.join(args.data_path, "ent2rel.json"), 'r')
ent2rel = json.load(f)
f.close()