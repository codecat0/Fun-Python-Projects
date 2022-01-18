# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dinomain.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import sys
import os
import random
import pygame

from setting import image_dir, music_dir, font_dir, screen_width, screen_height, screen, clock

# ----------------游戏参数设置------------ #
# 游戏速度
game_speed = 20
max_game_speed = 35
# 恐龙图像
dino_jump_img = pygame.image.load(os.path.join(image_dir, 'Dino/DinoJump.png'))
dino_dead_img = pygame.image.load(os.path.join(image_dir, 'Dino/DinoDead.png'))
dino_run_imgs = [
    pygame.image.load(os.path.join(image_dir, 'Dino/DinoRun1.png')),
    pygame.image.load(os.path.join(image_dir, 'Dino/DinoRun2.png'))
]
dino_down_imgs = [
    pygame.image.load(os.path.join(image_dir, 'Dino/DinoDuck1.png')),
    pygame.image.load(os.path.join(image_dir, 'Dino/DinoDuck2.png'))
]
# 背景图
ground_img = pygame.image.load(os.path.join(image_dir, 'Other/Track.png'))
# 背景中云图
cloud_img = pygame.image.load(os.path.join(image_dir, 'Other/Cloud.png'))
# 背景中树图
tree_imgs = [
    pygame.image.load(os.path.join(image_dir, 'Cactus/LargeCactus1.png')),
    pygame.image.load(os.path.join(image_dir, 'Cactus/LargeCactus2.png')),
    pygame.image.load(os.path.join(image_dir, 'Cactus/LargeCactus2.png')),
    pygame.image.load(os.path.join(image_dir, 'Cactus/SmallCactus1.png')),
    pygame.image.load(os.path.join(image_dir, 'Cactus/SmallCactus2.png')),
    pygame.image.load(os.path.join(image_dir, 'Cactus/SmallCactus3.png')),
]
# 背景中鸟图
bird_imgs = [
    pygame.image.load(os.path.join(image_dir, 'Bird/Bird1.png')),
    pygame.image.load(os.path.join(image_dir, 'Bird/Bird2.png'))
]
# 游戏结束时的图
game_over_img = pygame.image.load(os.path.join(image_dir, 'Other/GameOver.png'))
# 重新开始时的图
reset_img = pygame.image.load(os.path.join(image_dir, 'Other/Reset.png'))
# 跳跃时播发的音乐
jump_music = pygame.mixer.Sound(os.path.join(music_dir, 'big_jump.ogg'))
# 背景音乐
main_theme_music = os.path.join(music_dir, 'main_theme.ogg')
# 死亡时音乐
death_music = os.path.join(music_dir, 'death.wav')


class Ground:
    """背景的绘制"""

    def __init__(self):
        self.img = ground_img
        self.x = 0
        self.y = 400
        self.img_width = self.img.get_width()

    def draw(self, is_dead):
        screen.blit(ground_img, (self.x, self.y))
        screen.blit(ground_img, (self.x + self.img_width, self.y))
        if not is_dead:
            if self.x <= - self.img_width:
                self.x = 0
            self.x -= game_speed


class Cloud:
    """云的绘制"""

    def __init__(self):
        self.img = cloud_img
        self.rect = self.img.get_rect()
        self.rect.x = screen_width
        self.rect.y = random.randint(50, 300)

    def draw(self, is_dead):
        screen.blit(self.img, self.rect)
        if not is_dead:
            self.rect.x -= game_speed * 0.5


class Tree:
    """树的绘制"""

    def __init__(self):
        self.img = random.choice(tree_imgs)
        self.rect = self.img.get_rect()
        self.rect.x = screen_width
        self.rect.y = 425 - self.rect.height
        self.mask = pygame.mask.from_surface(self.img.convert_alpha())

    def draw(self, is_dead):
        screen.blit(self.img, self.rect)
        if not is_dead:
            self.rect.x -= game_speed


class Bird:
    """鸟的绘制"""

    def __init__(self):
        self.img = bird_imgs[0]
        self.rect = self.img.get_rect()
        self.rect.x = screen_width
        self.rect.y = random.randint(100, 400) - self.rect.height
        self.mask = pygame.mask.from_surface(self.img.convert_alpha())
        self.step = 0

    def draw(self, is_dead):
        self.img = bird_imgs[self.step // 5]
        rect = self.img.get_rect()
        rect.x = self.rect.x
        rect.y = self.rect.y
        self.rect = rect
        screen.blit(self.img, self.rect)
        if not is_dead:
            self.rect.x -= game_speed
            self.step = (self.step + 1) % 10


class Dino:
    """小恐龙动作的绘制"""

    def __init__(self):
        self.state = 0
        self.rect = dino_run_imgs[0].get_rect()
        self.rect.x = 88
        self.rect.y = 425 - self.rect.height
        self.mask = pygame.mask.from_surface(dino_run_imgs[0].convert_alpha())
        self.step = 0
        self.jump_speed = 8.5

    def draw(self):
        # 跑
        if self.state == 0:
            img = dino_run_imgs[self.step // 5]
            self.rect = img.get_rect()
            self.mask = pygame.mask.from_surface(img.convert_alpha())
            self.rect.x = 80
            self.rect.y = 425 - self.rect.height
        # 蹲
        elif self.state == 1:
            img = dino_down_imgs[self.step // 5]
            self.rect = img.get_rect()
            self.mask = pygame.mask.from_surface(img.convert_alpha())
            self.rect.x = 80
            self.rect.y = 425 - self.rect.height
        # 跳
        elif self.state == 2:
            img = dino_jump_img
            rect = img.get_rect()
            rect.x = 80
            rect.y = self.rect.y - (self.jump_speed * 4)
            self.jump_speed -= 0.8
            if self.jump_speed < -8.5:
                self.jump_speed = 8.5
                self.state = 0
                rect.y = 425 - dino_run_imgs[0].get_height()
                self.mask = pygame.mask.from_surface(dino_run_imgs[0].convert_alpha())
            else:
                self.mask = pygame.mask.from_surface(img.convert_alpha())
            self.rect = rect
        # 死亡
        else:
            img = dino_dead_img
            rect = img.get_rect()
            rect.x = self.rect.x
            rect.y = min(425 - img.get_height() + 10, self.rect.y)
            self.rect = rect
        screen.blit(img, self.rect)

    def update(self, keys):
        if self.state != 3:
            if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
                if self.state != 2:
                    jump_music.play()
                self.state = 2
            elif keys[pygame.K_DOWN] and self.state != 2:
                self.state = 1
            elif self.state != 2 and not keys[pygame.K_DOWN]:
                self.state = 0
        self.step = (self.step + 1) % 10


class Restart:
    """重新开始界面绘制"""

    def __init__(self):
        self.game_over_img = game_over_img
        self.reset_img = reset_img
        self.game_over_pos = (
            screen_width // 2 - self.game_over_img.get_width() // 2,
            screen_height // 4
        )
        self.reset_pos = (
            screen_width // 2 - self.reset_img.get_width() // 2,
            screen_height // 3
        )

    def draw(self, is_dead):
        if is_dead:
            screen.blit(self.game_over_img, self.game_over_pos)
            screen.blit(self.reset_img, self.reset_pos)


class Score:
    """分数绘制"""

    def __init__(self):
        self.score = 0
        self.font = pygame.font.Font(os.path.join(font_dir, 'SimHei.ttf'), 20)

    def draw(self, is_dead):
        if not is_dead:
            self.score += 1
            if self.score % 100 == 0:
                global game_speed
                game_speed = min(max_game_speed, game_speed + 1)

        score_str = str(self.score)
        score_str = "0" * max(5 - len(score_str), 0) + score_str
        if (self.score // 1000) % 2 == 0:
            text = self.font.render(score_str, True, (0, 0, 0))
        else:
            text = self.font.render(score_str, True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.x = screen_width - text_rect.width - 50
        text_rect.y = 50
        screen.blit(text, text_rect)


class DinoMain:
    def __init__(self):
        self.ground = Ground()
        self.cloud_list = []
        self.tree_and_bird_list = []
        self.dino = Dino()
        self.restart = Restart()
        self.score = Score()
        self.start = True
        self.play_music = 0

    def draw(self):
        if (self.score.score // 1000) % 2 == 0:
            screen.fill((255, 255, 255))
        else:
            screen.fill((0, 0, 0))

        self.ground.draw(self.dino.state == 3)
        for cloud in self.cloud_list:
            cloud.draw(self.dino.state == 3)
        for obj in self.tree_and_bird_list:
            obj.draw(self.dino.state == 3)
        self.dino.draw()
        self.score.draw(self.dino.state == 3)
        self.restart.draw(self.dino.state == 3)

    def main_loop(self):
        global game_speed

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        pygame.mixer.music.stop()
                        return
                # 按下任意按键重新开始游戏
                if self.dino.state == 3 and event.type == pygame.KEYDOWN:
                    self.ground = Ground()
                    self.cloud_list = []
                    self.tree_and_bird_list = []
                    self.dino = Dino()
                    self.restart = Restart()
                    self.score = Score()
                    self.start = True
                    game_speed = 20
                    pygame.time.delay(300)

            # 背景音乐播放设置
            if self.dino.state != 3 and self.play_music != 1:
                self.play_music = 1
                pygame.mixer.music.load(main_theme_music)
                pygame.mixer.music.play()
            elif self.dino.state == 3 and self.play_music != 2:
                self.play_music = 2
                pygame.mixer.music.load(death_music)
                pygame.mixer.music.play()

            if self.start:
                self.start = False
                continue
            else:
                keys = pygame.key.get_pressed()
                self.dino.update(keys)

            # 云的移动绘制
            new_cloud_list = []
            for cloud in self.cloud_list:
                if cloud.rect.x >= -cloud.rect.width:
                    new_cloud_list.append(cloud)
                else:
                    del cloud
            self.cloud_list = new_cloud_list
            if not len(self.cloud_list) or self.cloud_list[-1].rect.x <= (screen_width * 3 // 4):
                if random.random() > 0.8:
                    self.cloud_list.append(Cloud())

            # 障碍物（树和鸟）的绘制
            new_tree_and_bird_list = []
            for obj in self.tree_and_bird_list:
                if obj.rect.x >= -obj.rect.width:
                    new_tree_and_bird_list.append(obj)
                else:
                    del obj
            self.tree_and_bird_list = new_tree_and_bird_list
            if not len(self.tree_and_bird_list) or self.tree_and_bird_list[-1].rect.x <= (screen_width // 3):
                if random.random() > 0.8:
                    if random.random() > 0.7:
                        self.tree_and_bird_list.append(Bird())
                    else:
                        self.tree_and_bird_list.append(Tree())
            # 判断障碍物和小恐龙是否相碰撞
            for obj in self.tree_and_bird_list:
                offset = (self.dino.rect.x - obj.rect.x, self.dino.rect.y - obj.rect.y)
                if obj.mask.overlap(self.dino.mask, offset):
                    self.dino.state = 3

            self.draw()
            clock.tick(30)
            pygame.display.update()
