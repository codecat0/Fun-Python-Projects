# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : main.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import sys
import os
import pygame

from setting import image_dir, music_dir, font_dir, screen_width, screen_height, screen, clock
from dino import DinoMain

# 设置当前窗口的标题
pygame.display.set_caption('SmallDino')
# 从文件中加载dino跳跃图片
dino_jump_img = pygame.image.load(os.path.join(image_dir, 'Dino/DinoJump.png'))
# 背景图
ground_img = pygame.image.load(os.path.join(image_dir, 'Other/Track.png'))


class Menu:
    def __init__(self):
        self.start = 0
        self.start_finish = 0

        # 从系统字体库创建一个Font对象
        self.font = pygame.font.Font(os.path.join(font_dir, 'SimHei.ttf'), 30)
        # 绘制文本
        self.text = self.font.render('按下回车开始游戏', True, (0, 0, 0))
        self.text_rect = self.text.get_rect()
        self.text_rect.center = (screen_width // 2, screen_height // 2 - 100)

        self.dino_rect = dino_jump_img.get_rect()
        self.dino_rect.center = (screen_width // 2, screen_height // 2 - 200)
        self.jump_speed = 8.5

        # 空格表示跳、下箭头表示蹲下
        self.game_describle_texts = [self.font.render(t, True, (0, 0, 0)) for t in
                                     ['跳：空格', '蹲：下方向键']]
        self.game_describle_text_rects = [t.get_rect() for t in self.game_describle_texts]
        for i in range(len(self.game_describle_text_rects)):
            self.game_describle_text_rects[i].x = screen_width // 2 - 100
            self.game_describle_text_rects[i].y = screen_height // 2 + i * 50 - 15

        self.author_text = self.font.render('--- by ouyang & codecat', True, (0, 0, 0))
        self.author_text_rect = self.author_text.get_rect()
        self.author_text_rect.center = (screen_width // 2, screen_height // 2 + 150)

    def draw(self):
        screen.fill((255, 255, 255))
        screen.blit(self.text, self.text_rect)
        for i in range(len(self.game_describle_texts)):
            screen.blit(self.game_describle_texts[i], self.game_describle_text_rects[i])
        screen.blit(self.author_text, self.author_text_rect)
        screen.blit(ground_img, (screen_width // 2 - 60, screen_height // 2 - 180), (0, 0, 120, 70))

        if self.start != 0:
            self.dino_rect.centery -= self.jump_speed * 2
            self.jump_speed -= 0.8
            if self.jump_speed < -8.5:
                self.jump_speed = 8.5
                self.start = 0
                self.start_finish = 1
                self.dino_rect.center = (screen_width // 2, screen_height // 2 - 200)
        screen.blit(dino_jump_img, self.dino_rect)

    def menu_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_RETURN:
                        self.start = 1

            if self.start_finish:
                self.start = 0
                self.start_finish = 0
                main = DinoMain()
                main.main_loop()

            self.draw()
            clock.tick(30)
            pygame.display.update()


if __name__ == '__main__':
    menu = Menu()
    menu.menu_loop()
