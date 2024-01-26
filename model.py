import torch
import torch.nn as nn
from torch.nn import functional as F
import GLOBAL


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

        batch_size * 2 * 2, num_neighbor+1, hidden_size * 2 * 3
        """
        h = torch.matmul(inp, self.W)  # (B, N, out_features)
        N = h.shape[1]  # num_neighbor+1
        B = h.shape[0]  # batch_size*2

        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)  # (B, N, out_features)
        a_input = torch.cat((h, a), dim=2)  # (B, N, 2*out_features)

        # (B, N, 1)
        e = self.leakyrelu(torch.matmul(a_input, self.a))
        attention = F.softmax(e, dim=1)

        # mu是阈值，用于阻断较小的注意力值（认为其更可能是错误的）
        attention = attention - self.mu
        # 筛掉由于减mu出现的负值
        attention = (attention + abs(attention)) / 2.0

        # dropout
        attention = F.dropout(attention, self.dropout, training=self.training)

        attention = attention.reshape(B, 1, N)
        # (B, 1, N) * (B, N, out_features) => (B, 1, out_features)
        h_prime = torch.matmul(attention, h).squeeze(1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class BiLSTM_Attention(torch.nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_size = GLOBAL.args.BiLSTM_hidden_size
        self.num_layers = GLOBAL.args.BiLSTM_num_layers
        self.embedding_dim = GLOBAL.args.embedding_dim
        self.num_neighbor = GLOBAL.args.num_neighbor

        # 将实体和关系的特征维度映射为同一维度
        self.ent_mapping = nn.Linear(GLOBAL.node_embed.shape[-1], self.embedding_dim)
        self.rel_mapping = nn.Linear(GLOBAL.edge_weights.shape[-1], self.embedding_dim)
        nn.init.xavier_normal_(self.ent_mapping.weight)
        nn.init.xavier_normal_(self.rel_mapping.weight)

        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = GraphAttentionLayer(
            self.hidden_size * 2 * 3,
            self.hidden_size * 2 * 3,
            dropout=GLOBAL.args.dropout,
            alpha=GLOBAL.args.alpha,
            mu=GLOBAL.args.mu,
            concat=False,
        )

    def forward(self, batch_h, batch_r, batch_t):
        head = self.ent_mapping(batch_h).reshape(-1, self.embedding_dim)
        relation = self.rel_mapping(batch_r).reshape(-1, self.embedding_dim)
        tail = self.ent_mapping(batch_t).reshape(-1, self.embedding_dim)

        batch_triples = torch.stack([head, relation, tail]).reshape(-1, 3, self.embedding_dim)

        # 2 for bidirection
        h0 = torch.zeros(self.num_layers * 2, batch_triples.shape[0], self.hidden_size).to(GLOBAL.DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_triples.shape[0], self.hidden_size).to(GLOBAL.DEVICE)

        # Forward propagate LSTM
        # out: (batch_size * 2 * 2 * (num_neighbor+1), 3, hidden_size * 2)
        out, _ = self.lstm(batch_triples, (h0, c0))

        out = out.reshape(-1, self.hidden_size * 2 * 3)
        out = out.reshape(-1, self.num_neighbor + 1, self.hidden_size * 2 * 3)
        # (batch_size * 2 * 2, num_neighbor+1, hidden_size * 2 * 3)

        # (batch_size * 2 * 2, hidden_size * 2 * 3)
        out_att = self.attention(out)

        out = out.reshape(-1, self.num_neighbor * 2 + 2, self.hidden_size * 2 * self.seq_length)

        return out[:, 0, :], out_att
