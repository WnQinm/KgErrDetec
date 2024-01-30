import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import math
from get_batch import CompanyKgDataset, companyKgTrainDataCollator, companyKgTestDataCollator
import GLOBAL
from model import BiLSTM_Attention
import numpy as np

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from datetime import datetime
import os
import logging


def train(model_save_path):
    min_loss = np.inf
    iter_loss = []  # 每args.save_iter会计算平均值并清空

    dataset = CompanyKgDataset()
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=GLOBAL.args.batch_size, 
                            shuffle=True, 
                            num_workers=GLOBAL.args.num_workers, 
                            collate_fn=companyKgTrainDataCollator, 
                            pin_memory=True, 
                            drop_last=True, 
                            prefetch_factor=GLOBAL.args.prefetch_factor)
    model = BiLSTM_Attention(GLOBAL.node_embed.shape[-1], GLOBAL.edge_weights.shape[-1]).to(GLOBAL.DEVICE)
    if GLOBAL.args.train_existing_model:
        logging.info(f'continue training the model {GLOBAL.args.load_model_path}\n')
        model.load_state_dict(torch.load(GLOBAL.args.load_model_path))
    criterion = nn.MarginRankingLoss(GLOBAL.args.gama)
    optimizer = torch.optim.Adam(model.parameters(), lr=GLOBAL.args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=GLOBAL.args.T_max, eta_min=GLOBAL.args.eta_min)
    
    num_iterations = math.floor(GLOBAL.edges.shape[0] / GLOBAL.args.batch_size)
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
            scheduler.step()

            iter_loss.append(loss.detach().cpu().item())

            it += 1
            # 模型保存和日志打印
            if it % GLOBAL.args.save_iter == 0:
                iter_mean_loss = np.mean(iter_loss)
                f = open(model_save_path+'/loss.txt', 'a')
                for l in iter_loss:
                    f.write(str(l)+'\n')
                f.close()
                iter_loss = []
                logging.info(f'[Train] epoch {k+1}, iter {it}: mean loss {iter_mean_loss}')
                if iter_mean_loss < min_loss:
                    min_loss = iter_mean_loss
                    torch.save(model.state_dict(), model_save_path+'/best.ckpt')
                torch.save(model.state_dict(), model_save_path+'/latest.ckpt')
            pbar.update(1)


def test():
    model_load_path = GLOBAL.args.load_model_path
    num_iterations = math.floor(GLOBAL.edges.shape[0] / GLOBAL.args.batch_size)
    pbar = tqdm(total=num_iterations)

    dataset = CompanyKgDataset()
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=GLOBAL.args.batch_size, 
                            shuffle=True, 
                            num_workers=GLOBAL.args.num_workers, 
                            collate_fn=companyKgTestDataCollator, 
                            pin_memory=True, 
                            drop_last=True, 
                            prefetch_factor=GLOBAL.args.prefetch_factor)
    model = BiLSTM_Attention(GLOBAL.node_embed.shape[-1], GLOBAL.edge_weights.shape[-1]).to(GLOBAL.DEVICE)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ratios = torch.tensor(ratios, dtype=torch.float32, device=GLOBAL.DEVICE)
    num2ratios = torch.ceil(ratios * (GLOBAL.args.batch_size * 2)).int()

    with torch.no_grad():
        accs = None
        recalls = None
        it = 0

        for batch_h, batch_r, batch_t, labels in dataloader:
            batch_h = batch_h.to(GLOBAL.DEVICE)
            batch_r = batch_r.to(GLOBAL.DEVICE)
            batch_t = batch_t.to(GLOBAL.DEVICE)
            labels = labels.to(GLOBAL.DEVICE)

            out, out_att = model(batch_h, batch_r, batch_t)

            out_att = out_att.reshape(GLOBAL.args.batch_size * 2, 2, 2 * 3 * GLOBAL.args.BiLSTM_hidden_size)
            out_att_view0 = out_att[:, 0, :]
            out_att_view1 = out_att[:, 1, :]

            # 分数越小，说明嵌入得越好且三元组的两个视图的距离越近，置信度越高
            scores = GLOBAL.args.lam * torch.norm(out_att_view0 - out_att_view1, p=2, dim=1) + \
                     torch.norm(out[:, 0:2 * GLOBAL.args.BiLSTM_hidden_size] +
                                out[:, 2 * GLOBAL.args.BiLSTM_hidden_size:2 * 2 * GLOBAL.args.BiLSTM_hidden_size] -
                                out[:, 2 * 2 * GLOBAL.args.BiLSTM_hidden_size:2 * 3 * GLOBAL.args.BiLSTM_hidden_size], 
                                p=2, dim=1)
            
            # 获取scores从大到小排列的前batch_size个元素的索引
            _, top_indices = torch.topk(scores, GLOBAL.args.batch_size)
            pred = torch.cumsum(labels[top_indices], dim=0)[num2ratios-1]

            acc = (num2ratios - pred) / num2ratios
            recall = (num2ratios - pred) / GLOBAL.args.batch_size

            if accs is None:
                accs = acc.unsqueeze(dim=0)
            else:
                accs = torch.cat([accs, acc.unsqueeze(dim=0)], dim=0)

            if recalls is None:
                recalls = recall.unsqueeze(dim=0)
            else:
                recalls = torch.cat([recalls, recall.unsqueeze(dim=0)], dim=0)

            it += 1
            if it != 0 and it%GLOBAL.args.save_iter==0:
                mean_accs = torch.mean(accs, dim=0)
                mean_recalls = torch.mean(recalls, dim=0)
                logging.info(f'[Test] iter{it}/{num_iterations}:')
                for i in range(ratios.shape[0]):
                    logging.info(f'\tratio {ratios[i].detach().item():.2f}/0.5 acc {mean_accs[i].detach().item():.2f} recall {mean_recalls[i].detach().item():.2f}')
            pbar.update(1)
        
        mean_accs = torch.mean(accs, dim=0)
        mean_recalls = torch.mean(recalls, dim=0)
        logging.info('[Test] finish')
        for i in range(ratios.shape[0]):
            logging.info(f'\tratio {ratios[i].detach().item():.2f}/0.5 acc {mean_accs[i].detach().item():.2f} recall {mean_recalls[i].detach().item():.2f}')


def final():
    pass


if __name__ == '__main__':
    random.seed(GLOBAL.args.seed)
    torch.manual_seed(GLOBAL.args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(GLOBAL.args.seed)

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
    logging.info(f'{year}-{month}-{day} {str(datetime.now().time())} mode:{GLOBAL.args.mode}')

    with logging_redirect_tqdm():
        if GLOBAL.args.mode == 'train':
            print('-'*10)
            for k in GLOBAL.args.__dict__:
                logging.info(k + ": " + str(GLOBAL.args.__dict__[k]))
            print('-'*10)
            train(model_save_path)
        elif GLOBAL.args.mode == 'test':
            test()
