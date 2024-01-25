import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import logging

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, mu=0.001, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features 
        self.dropout = dropout 
        self.alpha = alpha 
        self.concat = concat
        self.mu = mu

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  

        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):
        """
        inp: input_fea [Batch_size, N, in_features]

        batch_size * 2 * 2, num_neighbor+1, hidden_size * 2 * seq_length
        """
        h = torch.matmul(inp, self.W)  # [batch_size, N, out_features]
        N = h.size()[1]  
        B = h.size()[0]  # B batch_size

        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)  # [batch_size, N, out_features]
        a_input = torch.cat((h, a), dim=2)  # [batch_size, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a))
        # [batch_size, N, 1] 

        attention = F.softmax(e, dim=1)  # [batch_size, N, 1]
        
        # mu是阈值，用于阻断较小的注意力值（认为其更可能是错误的）
        attention = attention - self.mu
        # 筛掉由于减mu出现的负值
        attention = (attention + abs(attention)) / 2.0

        # print(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        # print(attention)
        attention = attention.view(B, 1, N)
        h_prime = torch.matmul(attention, h).squeeze(1)  # [batch_size, 1, N]*[batch_size, N, out_features] => [batch_size, 1, out_features]
       
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class BiLSTM_Attention(torch.nn.Module):
    def __init__(self, args, input_size, hidden_size, num_layers, dropout, alpha, mu, device):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = 3
        self.BiLSTM_input_size = args.BiLSTM_input_size
        self.num_neighbor = args.num_neighbor
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.device = device
        self.attention = GraphAttentionLayer(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=alpha, mu=mu, concat=False)
        self.ent_embeddings = nn.Embedding(args.total_ent, args.embedding_dim)
        self.rel_embeddings = nn.Embedding(args.total_rel, args.embedding_dim)

        uniform_range = 6 / np.sqrt(args.embedding_dim)
        self.ent_embeddings.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_embeddings.weight.data.uniform_(-uniform_range, uniform_range)

    def forward(self, batch_h, batch_r, batch_t):
        head = self.ent_embeddings(batch_h)
        relation = self.rel_embeddings(batch_r)
        tail = self.ent_embeddings(batch_t)

        batch_triples_emb = torch.cat((head, relation), dim=1)
        batch_triples_emb = torch.cat((batch_triples_emb, tail), dim=1)

        # 极其抽象的代码，args里面实际上embed_dim = bilstm_input_dim，所以x形状是(batch_size, 3, embed_dim)
        x = batch_triples_emb.view(-1, 3, self.BiLSTM_input_size)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)# 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out:  (B, 3, hidden_size*2)

        # print('out_lstm', out_lstm.shape)

        # 它的第0维是batch_size*2*2*(num_neighbor+1)，待看数据预处理
        out = out.reshape(-1, self.hidden_size * 2 * self.seq_length)
        out = out.reshape(-1, self.num_neighbor + 1, self.hidden_size * 2 * self.seq_length)
        # [batch_size * 2 * 2, num_neighbor+1, dim_embedding] dim_embedding = hidden_size * 2 * seq_length

        out_att = self.attention(out)
        # [batch_size * 2 * 2, dim_embedding]

        out = out.reshape(-1, self.num_neighbor * 2 + 2, self.hidden_size * 2 * self.seq_length)
        return out[:, 0, :], out_att
