import torch
import torch.nn as nn
import random
import math
from get_batch import get_pair_batch
import GLOBAL
from model import BiLSTM_Attention

def train():
    seq = list(range(GLOBAL.edges.shape[0]))
    num_iterations = math.ceil(GLOBAL.edges.shape[0] / GLOBAL.args.batch_size)
    model = BiLSTM_Attention().to(GLOBAL.DEVICE)
    criterion = nn.MarginRankingLoss(GLOBAL.args.gama)
    optimizer = torch.optim.Adam(model.parameters(), lr=GLOBAL.args.learning_rate)
    # 进度条
    # logging

    for k in range(GLOBAL.args.max_epoch):
        for it in range(num_iterations):
            batch_h, batch_r, batch_t, batch_size = get_pair_batch(it, seq)

            out, out_att = model(batch_h, batch_r, batch_t)

            out = out.reshape(batch_size, -1, 2 * 3 * GLOBAL.args.BiLSTM_hidden_size)
            out_att = out_att.reshape(batch_size, -1, 2 * 3 * GLOBAL.args.BiLSTM_hidden_size)

            pos_h = out[:, 0, :]
            pos_z0 = out_att[:, 0, :]
            pos_z1 = out_att[:, 1, :]
            neg_h = out[:, 1, :]
            neg_z0 = out_att[:, 2, :]
            neg_z1 = out_att[:, 3, :]

            # loss function
            # positive
            pos_loss = GLOBAL.args.lam * torch.norm(pos_z0 - pos_z1, p=2, dim=1) + \
                       torch.norm(pos_h[:, 0:2 * GLOBAL.args.BiLSTM_hidden_size] +
                                  pos_h[:, 2 * GLOBAL.args.BiLSTM_hidden_size:2 * 2 * GLOBAL.args.BiLSTM_hidden_size] -
                                  pos_h[:, 2 * 2 * GLOBAL.args.BiLSTM_hidden_size:2 * 3 * GLOBAL.args.BiLSTM_hidden_size], 
                                  p=2, dim=1)
            # negative
            neg_loss = GLOBAL.args.lam * torch.norm(neg_z0 - neg_z1, p=2, dim=1) + \
                       torch.norm(neg_h[:, 0:2 * GLOBAL.args.BiLSTM_hidden_size] +
                                  neg_h[:, 2 * GLOBAL.args.BiLSTM_hidden_size:2 * 2 * GLOBAL.args.BiLSTM_hidden_size] -
                                  neg_h[:, 2 * 2 * GLOBAL.args.BiLSTM_hidden_size:2 * 3 * GLOBAL.args.BiLSTM_hidden_size], 
                                  p=2, dim=1)

            y = -torch.ones(batch_size).to(GLOBAL.DEVICE)
            loss = criterion(pos_loss, neg_loss, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def test():
    pass

if __name__ == '__main__':
    random.seed(GLOBAL.args.seed)
    torch.manual_seed(GLOBAL.args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(GLOBAL.args.seed)
    
    if GLOBAL.args.mode == 'train':
        train()
    elif GLOBAL.args.mode == 'test':
        test()