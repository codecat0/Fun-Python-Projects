# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import MovieDataset, movie_dataset_collate
from nets.model import Model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, optimizer, epoch, train_loader, val_loader, Epochs, cuda):
    total_loss = 0
    val_loss = 0
    model.train()
    print('Start Train')
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= len(train_loader):
                break

            usr, mov, score = batch

            with torch.no_grad():
                usr = [torch.from_numpy(var).long() for var in usr]
                mov = [torch.from_numpy(var).long() for var in mov]
                score_label = torch.from_numpy(score).type(torch.FloatTensor)

                if cuda:
                    usr = [var.cuda() for var in usr]
                    mov = [var.cuda() for var in mov]
                    score_label = score_label.cuda()

            optimizer.zero_grad()

            _, _, score_predict = model(usr, mov)
            loss = F.mse_loss(score_predict, score_label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)
    print('Finish Train')

    model.eval()
    print('Start Validation')
    with tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= len(val_loader):
                break
            usr, mov, score = batch

            with torch.no_grad():
                usr = [torch.from_numpy(var).long() for var in usr]
                mov = [torch.from_numpy(var).long() for var in mov]
                score_label = torch.from_numpy(score).type(torch.FloatTensor)

                if cuda:
                    usr = [var.cuda() for var in usr]
                    mov = [var.cuda() for var in mov]
                    score_label = score_label.cuda()

            _, _, score_predict = model(usr, mov)
            loss = F.mse_loss(score_predict, score_label)

            val_loss += loss.item()

            pbar.set_postfix(**{
                'total_loss': val_loss / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epochs))
    print('Total loss %.3f || Val loss %.3f' % (total_loss / len(train_loader), val_loss / len(val_loader)))


if __name__ == '__main__':
    # 是否使用GPU训练、batch_size大小、学习率，多线程读取数据、是否使用海报数据
    cuda = True
    batch_size = 512
    lr = 0.001
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    use_poster = False
    Epochs = 100
    # 定义模型
    model = Model(use_poster=use_poster)
    if cuda:
        model.cuda()
    # 优化器及学习率的变化
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    # 训练集
    train_dataset = MovieDataset(
        use_poster=use_poster,
        mode='train'
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=movie_dataset_collate
    )
    # 验证集
    val_dataset = MovieDataset(
        use_poster=use_poster,
        mode='valid'
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=movie_dataset_collate
    )
    for epoch in range(Epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            Epochs=Epochs,
            cuda=cuda
        )
        lr_scheduler.step()

    # 保存模型
    torch.save(model.state_dict(), 'weights/net.pth')