import pygame
import numpy as np
import WeaponSystem as ws

class Actor:
    def __init__(self, settings, spritePath):
        self.type = "not set"
        self.settings = settings

        #Physical attributes
        self.pos = np.array([settings.screenRes / 2, settings.screenRes / 5])
        self.oldPos = self.pos.copy()
        self.speed = 10
        self.ws = ws.WeaponSystem(settings, spritePath)
        self.size = settings.screenRes // 25
        # needed for collision checking
        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.size]))
        self.vertices.append(np.array([self.size, 0]))
        self.vertices.append(np.array([self.size, self.size]))

        #Graphical attributes
        self.sprite = pygame.image.load(spritePath)
        self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))