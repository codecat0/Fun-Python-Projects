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
    print(dataset.max_usr_age)
    print(dataset.max_usr_job)
    print(dataset.max_mov_cat)
    print(dataset.max_mov_tit)
