import numpy as np
import pygame

class Bullet:
    def __init__(self, settings):
        self.settings = settings

        self.pos = np.array([0, 0])
        self.size = 16
        self.dir = np.array([0, 0])
        self.speed = 15
        self.timer = 0
        self.TTL = 3 #time to live in seconds

        self.h = settings.screenHeight / 100
        self.w = settings.screenWidth / 100
        self.sprite = pygame.image.load("./Assets/Player.png")
        self.sprite = pygame.transform.scale(self.sprite, (self.w, self.h))

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
        self.pos = np.array(pos) - int(self.size/2)
        self.dir = np.array(dir)
        self.timer = 0

    def update(self):
        if (self.active):
            self.timer += 1
            self.pos += self.dir * self.speed
            if self.timer >= self.TTL * self.settings.gameSecond:
                self.active = False

