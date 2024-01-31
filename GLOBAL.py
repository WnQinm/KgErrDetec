import torch
import argparse
import os
import json

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--mode', default='train', choices=['train', 'test'], help='run training or evaluation')
parser.add_argument('--save_dir', default=f'./checkpoints/', help='训练时模型的保存路径')
parser.add_argument('--load_model_path', default=f'./latest.ckpt', help='加载模型的路径, 要精确到ckpt文件')
parser.add_argument('--save_iter', default=500, help='在训练时每多少iter打印一次损失并保存checkpoint; 在测试时每多少iter打印一次累计至当前iter的平均acc和recall')
parser.add_argument('--num_workers', default=15, help='同时启用多个进程在cpu加载数据, 以平衡加载数据和训练数据的速度差距, 推荐为cpu物理核心数-1或-2')
parser.add_argument('--prefetch_factor', default=10, help='每个进程预加载的batch数量, 配合num_workers尽可能保证gpu不空闲')
parser.add_argument('--train_existing_model', default=True, type=bool, help='是否继续训练之前训练过的模型, 模型路径为args.load_model_path')

# data
parser.add_argument('--data_path', default=f'./', help='path to the dataset')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')

# model architecture
parser.add_argument('--BiLSTM_hidden_size', default=32, type=int, help='BiLSTM hidden size, 模型输出的特征维度将是hidden_size*2*3')
parser.add_argument('--BiLSTM_num_layers', default=2, type=int, help='BiLSTM layers')
parser.add_argument('--num_neighbor', default=39, type=int, help='number of neighbors')
parser.add_argument('--embedding_dim', default=32, type=int, help='实体和关系需要嵌入到同一维度')

# regularization
parser.add_argument('--alpha', type=float, default=0.2, help='leakyrelu hyperparameter alpha')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout for EaGNN')

# optimization
parser.add_argument('--max_epoch', default=1, help='max epochs')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--gama', default=0.5, type=float, help="margin parameter(MarginRankingLoss)")
parser.add_argument('--lam', default=0.1, type=float, help="嵌入损失和对比损失的比例")
parser.add_argument('--mu', default=0.001, type=float, help="在GAT中阻断小于mu的注意力值(认为其更可能是错误的三元组故而筛掉)")

# lr scheduler
parser.add_argument('--T_max', default=790000, type=int, help="余弦退火半个周期数")
parser.add_argument('--eta_min', default=1e-5, type=float, help="余弦退火最小值(最大值为args.learning_rate)")

args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('loading dataset...')
# edges: (edge_num, 2)
# edge_weights: (edge_num, 15)
# node_embed(PAUSE): (node_num, 32)
print('1/4 loading edges         ', end='')
edges = torch.load(os.path.join(args.data_path, "edges.pt")).to(torch.int32)
print('√')

print('2/4 loading edges weight  ', end='')
edge_weights = torch.load(os.path.join(args.data_path, "edges_weight.pt")).to_dense().to(torch.float32)
print('√')

print('3/4 loading nodes feature ', end='')
node_embed = torch.load(os.path.join(args.data_path, "nodes_feature_pause.pt")).to(torch.float32)
print('√')

print('4/4 loading ent2rel       ', end='')
f = open(os.path.join(args.data_path, "ent2rel.json"), 'r')
ent2rel = json.load(f)
f.close()
print('√\n')
