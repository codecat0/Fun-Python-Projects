# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataset import MovieDataset


class Model(nn.Module):
    def __init__(self, use_poster,  fc_sizes=(128, 64, 32)):
        super(Model, self).__init__()
        self.use_poster = use_poster

        dataset = MovieDataset(use_poster=use_poster)

        usr_gender_dict_size = 2
        self.usr_gender_emb = nn.Embedding(num_embeddings=usr_gender_dict_size, embedding_dim=16)
        self.usr_gender_fc = nn.Linear(in_features=16, out_features=16)

        usr_age_dict_size = dataset.max_usr_age + 1
        self.usr_age_emb = nn.Embedding(num_embeddings=usr_age_dict_size, embedding_dim=16)
        self.usr_age_fc = nn.Linear(in_features=16, out_features=16)

        usr_job_dict_size = dataset.max_usr_job + 1
        self.usr_job_emb = nn.Embedding(num_embeddings=usr_job_dict_size, embedding_dim=16)
        self.usr_job_fc = nn.Linear(in_features=16, out_features=16)

        self.usr_combined = nn.Linear(in_features=48, out_features=200)

        mov_cat_dict_size = dataset.max_mov_cat + 1
        self.mov_cat_emb = nn.Embedding(num_embeddings=mov_cat_dict_size, embedding_dim=32)
        self.mov_cat_fc = nn.Linear(in_features=32, out_features=32)

        mov_tit_dict_size = dataset.max_mov_tit + 1
        self.mov_tit_emb = nn.Embedding(num_embeddings=mov_tit_dict_size, embedding_dim=32)
        # (b, 1, 15, 32) -> (b, 1, 7, 32)
        self.mov_tit_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(2, 1))
        # (b, 1, 7, 32) -> (b, 1, 5, 32)
        self.mov_tit_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1))

        self.mov_poster_conv = nn.Sequential(
            # (b, 3, 64, 64) -> (b, 32, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (b, 32, 32, 32) -> (b, 32, 16, 16)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (b, 32, 16, 16) -> (b, 64, 8, 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (b, 64, 8, 8) -> (b, 64, 1, 1)
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.mov_poster_fc = nn.Linear(in_features=64, out_features=32)

        if use_poster:
            self.mov_combined = nn.Linear(in_features=96, out_features=200)
        else:
            self.mov_combined = nn.Linear(in_features=64, out_features=200)

        # (b, 200) -> (b, 128) -> (b, 64) -> (b, 32)
        self.usr_layers = self.make_fc_layers(fc_sizes)
        self.mov_layers = self.make_fc_layers(fc_sizes)

    @staticmethod
    def make_fc_layers(fc_sizes):
        layers = []
        in_channels = 200
        for v in fc_sizes:
            layers += [nn.Linear(in_channels, v), nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def get_usr_feat(self, usr_var):
        usr_gender, usr_age, usr_job = usr_var
        feats_collect = []
        # 性别特征提取
        usr_gender = self.usr_gender_emb(usr_gender)
        usr_gender = self.usr_gender_fc(usr_gender)
        usr_gender = F.relu(usr_gender)
        feats_collect.append(usr_gender)
        # 年龄特征提取
        usr_age = self.usr_age_emb(usr_age)
        usr_age = self.usr_age_fc(usr_age)
        usr_age = F.relu(usr_age)
        feats_collect.append(usr_age)
        # 职业特征提取
        usr_job = self.usr_job_emb(usr_job)
        usr_job = self.usr_job_fc(usr_job)
        usr_job = F.relu(usr_job)
        feats_collect.append(usr_job)

        usr_feat = torch.cat(feats_collect, dim=1)
        usr_feat = torch.tanh(self.usr_combined(usr_feat))
        return usr_feat

    def get_mov_feat(self, mov_var):
        mov_cat, mov_title, mov_poster = mov_var
        feats_collect = []
        batch_size = mov_cat.shape[0]
        # 电影类别特征提取
        mov_cat = self.mov_cat_emb(mov_cat)
        mov_cat = torch.sum(mov_cat, dim=1, keepdim=False)
        mov_cat = self.mov_cat_fc(mov_cat)
        mov_cat = F.relu(mov_cat)
        feats_collect.append(mov_cat)
        # 电影名称特征提取
        mov_title = self.mov_tit_emb(mov_title)
        mov_title = F.relu(self.mov_tit_conv2(F.relu(self.mov_tit_conv(mov_title))))
        mov_title = torch.sum(mov_title, dim=2, keepdim=False)
        mov_title = F.relu(mov_title)
        mov_title = mov_title.reshape(batch_size, -1)
        feats_collect.append(mov_title)
        # 电影海报特征提取
        if self.use_poster:
            mov_poster = mov_poster.float()
            mov_poster = self.mov_poster_conv(mov_poster)
            mov_poster = mov_poster.reshape(batch_size, -1)
            mov_poster = self.mov_poster_fc(mov_poster)
            feats_collect.append(mov_poster)

        mov_feat = torch.cat(feats_collect, dim=1)
        mov_feat = torch.tanh(self.mov_combined(mov_feat))
        return mov_feat

    def forward(self, usr_var, mov_var):
        usr_feat = self.get_usr_feat(usr_var)
        mov_feat = self.get_mov_feat(mov_var)

        for fc_layer in self.usr_layers:
            usr_feat = fc_layer(usr_feat)

        for fc_layer in self.mov_layers:
            mov_feat = fc_layer(mov_feat)

        sim = F.cosine_similarity(usr_feat, mov_feat) * 5
        return usr_feat, mov_feat, sim