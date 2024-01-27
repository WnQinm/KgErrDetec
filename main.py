import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import math
from get_batch import CompanyKgDataset, companyKgDataCollator
import GLOBAL
from model import BiLSTM_Attention
import numpy as np

from tqdm import tqdm
from datetime import datetime
import os
import logging


def train():
    year = str(datetime.now().year)
    month = str(datetime.now().month)
    month = '0'+month if len(month)==1 else month
    day = str(datetime.now().day)
    day = '0'+day if len(day)==1 else day
    model_save_path = os.path.join(GLOBAL.args.save_dir + f'{year+month+day}')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(model_save_path, 'log.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    
    min_loss = np.inf
    iter_loss = []  # 每args.save_iter会计算平均值并清空

    dataset = CompanyKgDataset()
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=GLOBAL.args.batch_size, 
                            shuffle=True, 
                            num_workers=22, 
                            collate_fn=lambda b: companyKgDataCollator(b, dataset), 
                            pin_memory=True, 
                            drop_last=True)
    num_iterations = math.ceil(dataset.edges.shape[0] / GLOBAL.args.batch_size)
    model = BiLSTM_Attention(dataset.node_embed.shape[-1], dataset.edge_weights.shape[-1]).to(GLOBAL.DEVICE)
    criterion = nn.MarginRankingLoss(GLOBAL.args.gama)
    optimizer = torch.optim.Adam(model.parameters(), lr=GLOBAL.args.learning_rate)
    
    pbar = tqdm(total=GLOBAL.args.max_epoch*num_iterations)
    for k in range(GLOBAL.args.max_epoch):
        it = 0
        for batch_h, batch_r, batch_t in dataloader:

            batch_h = batch_h.to(GLOBAL.DEVICE)
            batch_r = batch_r.to(GLOBAL.DEVICE)
            batch_t = batch_t.to(GLOBAL.DEVICE)

            out, out_att = model(batch_h, batch_r, batch_t)

            out = out.reshape(-1, 2, 2 * 3 * GLOBAL.args.BiLSTM_hidden_size)
            out_att = out_att.reshape(-1, 4, 2 * 3 * GLOBAL.args.BiLSTM_hidden_size)
            
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

            y = -torch.ones(GLOBAL.args.batch_size).to(GLOBAL.DEVICE)
            
            loss = criterion(pos_loss, neg_loss, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_loss.append(loss.detach().cpu().item())
            # 模型保存和日志打印
            if it > 0 and it % GLOBAL.args.save_iter == 0:
                iter_mean_loss = np.mean(iter_loss)
                f = open(model_save_path+'/loss.txt', 'w')
                for l in iter_loss:
                    f.write(str(l)+'\n')
                f.close()
                iter_loss = []
                logging.info(f'epoch {k}, iter {it}: mean loss {iter_mean_loss}')
                if iter_mean_loss < min_loss:
                    min_loss = iter_mean_loss
                    torch.save(model.state_dict(), model_save_path+'/best.ckpt')
                torch.save(model.state_dict(), model_save_path+'/latest.ckpt')
            pbar.update(1)
            it += 1


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