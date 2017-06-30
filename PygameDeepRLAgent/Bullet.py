import numpy as np
import pygame

class Bullet:
    def __init__(self, settings, sprite):
        self.settings = settings

        self.type = "bullet"

        self.pos = np.array([0, 0])
        self.oldPos = self.pos.copy()
        self.dir = np.array([0, 0])
        self.speed = 15
        self.timer = 0
        self.TTL = 3 #time to live in seconds

        self.h = int(settings.screenHeight / 40)
        self.w = int(settings.screenWidth / 40)
        self.size = [self.w, self.h]
        self.sprite = pygame.image.load(sprite)
        self.sprite = pygame.transform.scale(self.sprite, (self.w, self.h))

        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.h]))
        self.vertices.append(np.array([self.w, 0]))
        self.vertices.append(np.array([self.w, self.h]))
        self.active = False

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)

    def shoot(self, pos, dir):
        self.active = True
        self.pos = np.array(pos)- np.array([self.w/2, self.h/2])
        self.dir = np.array(dir)
        self.timer = 0

    def update(self):
        if (self.active):
            self.timer += 1
            self.oldPos = self.pos.copy()
            self.pos += self.dir * self.speed
            if self.timer >= self.TTL * self.settings.gameSecond:
                self.active = False

