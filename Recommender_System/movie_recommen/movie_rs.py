# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : movie_rs.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from PIL import Image
from nets.model import Model
from dataset.dataset import MovieDataset


def get_usr_mov_features(model, dataset, params_file_path, poster_path, use_poster):
    """
    获取用户特征和电影特征
    :param model: 模型
    :param dataset: 数据集
    :param params_file_path: 模型参数路径
    :param poster_path: 电影海报路径
    :param use_poster: 是否使用电影海报来构建特征
    """
    usr_pkl = {}
    mov_pkl = {}

    def list2tensor(inputs, shape):
        inputs = np.reshape(np.array(inputs).astype(np.int64), shape)
        return torch.from_numpy(inputs)

    model_state_dict = torch.load(params_file_path)
    model.load_state_dict(model_state_dict)
    model.eval()

    dataset = dataset.dataset

    for i in range(len(dataset)):
        usr_info, mov_info, score = dataset[i]['usr_info'], dataset[i]['mov_info'], dataset[i]['scores']
        usr_id = str(usr_info['usr_id'])
        mov_id = str(mov_info['mov_id'])

        # 获得用户数据，计算得到用户特征，保存在usr_pkl字典中
        if usr_id not in usr_pkl.keys():
            usr_gender_v = list2tensor(usr_info['gender'], [1])
            usr_age_v = list2tensor(usr_info['age'], [1])
            usr_job_v = list2tensor(usr_info['job'], [1])

            usr_in = [usr_gender_v, usr_age_v, usr_job_v]
            usr_feat = model.get_usr_feat(usr_in)

            usr_pkl[usr_id] = usr_feat.detach().numpy()

        if mov_id not in mov_pkl.keys():
            mov_tit_v = list2tensor(mov_info['title'], [1, 1, 15])
            mov_cat_v = list2tensor(mov_info['category'], [1, 6])
            poster_v = None
            if use_poster:
                mov_id = mov_info['mov_id']
                poster = Image.open(poster_path + 'mov_id{}.jpg'.format(str(mov_id)))
                poster = poster.resize((64, 64))
                poster = np.array(poster) / 127.5 - 1
                poster_v = list2tensor(poster, [1, 3, 64, 64])
            mov_in = [mov_cat_v, mov_tit_v, poster_v]
            mov_feat = model.get_mov_feat(mov_in)

            mov_pkl[mov_id] = mov_feat.detach().numpy()

    pickle.dump(usr_pkl, open('weights/usr_feat.pkl', 'wb'))
    pickle.dump(mov_pkl, open('weights/mov_feat.pkl', 'wb'))
    print('usr and movie features saved!!!')


def recommend_mov_for_usr(usr_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path):
    """
    根据用户兴趣推荐电影
    :param usr_id: 用户id
    :param top_k: 最感兴趣的电影数
    :param pick_num: 在topk中选择多少个
    :param usr_feat_dir: 用户特征目录
    :param mov_feat_dir: 电影特征目录
    :param mov_info_path: 电影信息目录
    """
    assert pick_num <= top_k
    # 读取电影和用户的特征
    usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
    mov_feats = pickle.load(open(mov_feat_dir, 'rb'))
    usr_feat = usr_feats[str(usr_id)]
    usr_feat = torch.from_numpy(usr_feat)
    cos_sims = []
    for idx, key in enumerate(mov_feats.keys()):
        mov_feat = mov_feats[key]
        mov_feat = torch.from_numpy(mov_feat)
        sim = F.cosine_similarity(usr_feat, mov_feat)
        # 计算余弦相似度
        cos_sims.append(sim.numpy()[0])
    # 对相似度排序
    index = np.argsort(cos_sims)[-top_k:]

    mov_info = {}
    with open(mov_info_path, 'r', encoding='ISO-8859-1') as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split('::')
            mov_info[str(item[0])] = item

    print('usr_id:', usr_id)
    print('推荐可能喜欢的电影是：')
    res = []
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)

    for mov_id in res:
        print('mov_id:', mov_id, mov_info[str(mov_id)])


def get_usr_rating_topk(usr_a, top_k, rating_path, mov_info_path):
    with open(rating_path, 'r') as f:
        ratings_data = f.readlines()

    usr_rating_info = {}
    for item in ratings_data:
        item = item.strip().split('::')
        usr_id, mov_id, score = item[0], item[1], item[2]
        if usr_id == str(usr_a):
            usr_rating_info[mov_id] = float(score)

    # 获得评分过的电影ID
    movie_ids = list(usr_rating_info.keys())
    print('id为 {} 的用户，评分过的电影数为：{}'.format(usr_a, len(movie_ids)))

    rating_topk = sorted(usr_rating_info.items(), key=lambda x: x[1])[-top_k:]

    mov_info = {}
    with open(mov_info_path, 'r', encoding='ISO-8859-1') as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split('::')
            mov_info[str(item[0])] = item

    for k, score in rating_topk:
        print('电影id：{}， 评分是：{}，电影信息是：{}'.format(k, score, mov_info[k]))




if __name__ == '__main__':
    use_poster = False
    model = Model(use_poster=use_poster)
    dataset = MovieDataset(use_poster=use_poster)
    params_file_path = './weights/net.pth'
    poster_path = './data/posters/'
    get_usr_mov_features(
        model=model,
        dataset=dataset,
        params_file_path=params_file_path,
        poster_path=poster_path,
        use_poster=use_poster
    )

    usr_id = 2
    top_k, pick_num = 10, 6
    usr_feat_dir = './weights/usr_feat.pkl'
    mov_feat_dir = './weights/mov_feat.pkl'
    mov_info_path = './data/movies.dat'
    recommend_mov_for_usr(
        usr_id=usr_id,
        top_k=top_k,
        pick_num=pick_num,
        usr_feat_dir=usr_feat_dir,
        mov_feat_dir=mov_feat_dir,
        mov_info_path=mov_info_path
    )

    usr_a = 2
    rating_path = './data/ratings.dat'
    get_usr_rating_topk(
        usr_a=usr_a,
        top_k=top_k,
        rating_path=rating_path,
        mov_info_path=mov_info_path
    )