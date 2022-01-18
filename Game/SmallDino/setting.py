# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : setting.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import pygame
pygame.init()

image_dir = '/home/hlz/PycharmProjects/Fun_Projects/SmallDino/Image'
music_dir = '/home/hlz/PycharmProjects/Fun_Projects/SmallDino/Music'
font_dir = '/home/hlz/PycharmProjects/Fun_Projects/SmallDino/Fonts'

screen_width, screen_height = 1400, 700
screen = pygame.display.set_mode(size=(screen_width, screen_height))
clock = pygame.time.Clock()