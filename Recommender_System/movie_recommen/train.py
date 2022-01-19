# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from dataset.dataset import MovieDataset

if __name__ == '__main__':
    dataset = MovieDataset(use_poster=True)
    usr, mov, score = dataset[0]
    print(usr)
    for v in mov:
        print(v.shape)
    print(score)
