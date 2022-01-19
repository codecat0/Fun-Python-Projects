# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dataset.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, use_poster, mode='train'):
        self.use_poster = use_poster
        assert mode in ['train', 'valid'], 'mode must in [train, valid]'
        self.mode = mode
        usr_info_path = './data/users.dat'
        movie_info_path = './data/movies.dat'
        if use_poster:
            rating_path = './data/new_rating.txt'
        else:
            rating_path = './data/ratings.dat'
        self.poster_path = './data/posters/'

        # 获得电影数据
        self.movie_info, self.movie_cat, self.movie_title = self.get_movie_info(movie_info_path)
        # 记录电影的最大ID
        self.max_mov_cat = np.max([self.movie_cat[k] for k in self.movie_cat.keys()])
        self.max_mov_tit = np.max([self.movie_title[k] for k in self.movie_title.keys()])
        self.max_mov_id = np.max(list(map(int, self.movie_info.keys())))
        # 记录用户数据的最大ID
        self.max_usr_id = 0
        self.max_usr_age = 0
        self.max_usr_job = 0
        # 得到用户数据
        self.usr_info = self.get_usr_info(usr_info_path)
        # 得到评分数据
        self.rating_info = self.get_rating_info(rating_path)
        # 构建数据集
        self.dataset = self.get_dataset(
            usr_info=self.usr_info,
            rating_info=self.rating_info,
            movie_info=self.movie_info
        )
        # 划分数据集
        self.train_dataset = self.dataset[:int(len(self.dataset) * 0.9)]
        self.valid_dataset = self.dataset[int(len(self.dataset) * 0.9):]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset)
        else:
            return len(self.valid_dataset)

    def __getitem__(self, idx):
        if self.mode == 'train':
            dataset = self.train_dataset
        else:
            dataset = self.valid_dataset

        # usr_id = dataset[idx]['usr_info']['usr_id']
        # usr_id = np.array(usr_id)
        gender = dataset[idx]['usr_info']['gender']
        gender = np.array(gender)
        age = dataset[idx]['usr_info']['age']
        age = np.array(age)
        job = dataset[idx]['usr_info']['job']
        job = np.array(job)

        mov_id = dataset[idx]['mov_info']['mov_id']
        title = dataset[idx]['mov_info']['title']
        title = np.reshape(np.array(title), [1, 15]).astype(np.int64)
        category = dataset[idx]['mov_info']['category']
        category = np.array(category)

        if self.use_poster:
            poster = Image.open(self.poster_path + 'mov_id{}.jpg'.format(str(mov_id)))
            poster = poster.resize((64, 64))
            if len(poster.size) <= 2:
                poster = poster.convert('RGB')

            poster = np.reshape(np.array(poster) / 127.5 - 1, [3, 64, 64]).astype(np.float32)
        else:
            poster = np.array([0.])

        score = int(dataset[idx]['scores'])
        score = np.array(score)

        return [gender, age, job], [category, title, poster], score

    @staticmethod
    def get_dataset(usr_info, rating_info, movie_info):
        dataset_list = []
        for usr_id in rating_info.keys():
            usr_ratings = rating_info[usr_id]
            for movie_id in usr_ratings.keys():
                dataset_list.append(
                    {
                        'usr_info': usr_info[usr_id],
                        'mov_info': movie_info[movie_id],
                        'scores': usr_ratings[movie_id]
                    }
                )
        return dataset_list

    @staticmethod
    def get_movie_info(path):
        """获取电影数据"""
        with open(path, 'r', encoding="ISO-8859-1") as f:
            data = f.readlines()
        # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
        movie_info, movie_titles, movie_cat = {}, {}, {}
        # 对电影名字、类别中不同的单词计数(不同的单词用对应的数字序号指代)
        t_count, c_count = 1, 1

        for item in data:
            item = item.strip().split('::')
            v_id = item[0]
            v_title = item[1][:-7]
            v_year = item[1][-5:-1]
            cats = item[2].split('|')

            titles = v_title.split()

            # 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
            for t in titles:
                if t not in movie_titles:
                    movie_titles[t] = t_count
                    t_count += 1

            # 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
            for cat in cats:
                if cat not in movie_cat:
                    movie_cat[cat] = c_count
                    c_count += 1

            # 补0使电影名称对应所有名称中最长的
            v_tit = [movie_titles[k] for k in titles]
            while len(v_tit) < 15:
                v_tit.append(0)

            # 补0使电影种类对应所有电影类别最多的
            v_cat = [movie_cat[k] for k in cats]
            while len(v_cat) < 6:
                v_cat.append(0)

            movie_info[v_id] = {
                'mov_id': int(v_id),
                'title': v_tit,
                'category': v_cat,
                'years': int(v_year)
            }

        return movie_info, movie_cat, movie_titles

    def get_usr_info(self, path):
        def gender2num(gender):
            """性别转换"""
            return 1 if gender == 'F' else 0

        with open(path, 'r') as f:
            data = f.readlines()

        # 用户信息的字典
        usr_info = {}
        for item in data:
            item = item.strip().split('::')
            usr_id = item[0]
            usr_info[usr_id] = {
                'usr_id': int(usr_id),
                'gender': gender2num(item[1]),
                'age': int(item[2]),
                'job': int(item[3])
            }
            self.max_usr_id = max(self.max_mov_id, int(usr_id))
            self.max_usr_age = max(self.max_usr_age, int(item[2]))
            self.max_usr_job = max(self.max_usr_job, int(item[3]))

        return usr_info

    @staticmethod
    def get_rating_info(path):
        """得到评分数据"""
        with open(path, 'r') as f:
            data = f.readlines()

        rating_info = {}
        for item in data:
            item = item.strip().split('::')
            usr_id, movie_id, score = item[0], item[1], item[2]
            if usr_id not in rating_info:
                rating_info[usr_id] = {movie_id: float(score)}
            else:
                rating_info[usr_id][movie_id] = float(score)

        return rating_info


def movie_dataset_collate(batch):
    usr_gender_list = []
    usr_age_list = []
    usr_job_list = []

    mov_cat_list = []
    mov_tit_list = []
    mov_poster_list = []

    score_list = []

    for usr, mov, score in batch:
        for gender, age, job in usr:
            usr_gender_list.append(gender)
            usr_age_list.append(age)
            usr_job_list.append(job)

        for cat, tit, poster in mov:
            mov_cat_list.append(cat)
            mov_tit_list.append(tit)
            mov_poster_list.append(poster)

        score_list.append(score)

    return [np.array(usr_gender_list), np.array(usr_age_list), np.array(usr_job_list)], \
           [np.array(mov_cat_list), np.array(mov_tit_list), np.array(mov_poster_list)], np.array(score_list)
