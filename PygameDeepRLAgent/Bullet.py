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

        self.size = settings.screenRes // 40
        self.sprite = pygame.image.load(sprite)
        self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))

        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.size]))
        self.vertices.append(np.array([self.size, 0]))
        self.vertices.append(np.array([self.size, self.size]))
        self.active = False

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)

    def shoot(self, pos, dir):
        self.active = True
        self.pos = np.array(pos)- np.array([self.size/2, self.size/2])
        self.dir = np.array(dir)
        self.timer = 0

    def update(self):
        if (self.active):
            self.timer += 1
            self.oldPos = self.pos.copy()
            self.pos += self.dir * self.speed
            if self.timer >= self.TTL * self.settings.gameSecond:
                self.active = False

