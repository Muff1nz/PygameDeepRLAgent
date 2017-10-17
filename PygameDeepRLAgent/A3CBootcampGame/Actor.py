import numpy as np
import pygame

class Actor:
    def __init__(self, settings, spritePath, size):
        self.settings = settings
        self.type = "not set"
        self.collisionGroup = -1
        self.timeStep = -1
        self.active = True

        #Physical attributes
        self.pos = np.array([settings.gameRes / 2, settings.gameRes / 2])
        self.oldPos = self.pos.copy()
        self.size = int(settings.gameRes * size)
        # needed for collision checking
        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.size]))
        self.vertices.append(np.array([self.size, 0]))
        self.vertices.append(np.array([self.size, self.size]))

        #Graphical attributes
        self.sprite = pygame.image.load(spritePath)
        self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))

    def onBoxCollision(self, other):
        self.active = False

    def onWallCollision(self):
        pass