import pygame
import numpy as np
import random

from Actor import Actor

class Enemy(Actor):
    def __init__(self, settings, spritePath, world):
        super(Enemy, self).__init__(settings, spritePath)
        self.type = "enemy"
        #AI attributes
        self.active = True
        self.nodes = world.nodes
        self.spawn()

    def update(self):
        #Move to trarget node, and choose a new when reached
        if self.active:
            if np.linalg.norm(self.targetNode.pos - self.pos) < 40:
                self.centerOnNode(self.targetNode)
                self.targetNode = self.targetNode.neighbors[random.randint(0, len(self.targetNode.neighbors) - 1)]
                self.calcDir()
            self.oldPos = self.pos.copy()
            self.pos += self.dir * self.speed
            #Shoot in a random direction
            shootdDir = [0, 0]
            sample = [1, -1]
            shootdDir[random.randint(0, 1)] = sample[random.randint(0, 1)]
            self.ws.shoot(shootdDir, self.pos + self.size / 2)
            self.ws.update()
        else:
            if self.ws.active:
                self.ws.update()

    def spawn(self):
        index = random.randint(0, len(self.nodes) - 1)
        self.targetNode = self.nodes[index]
        self.centerOnNode(self.targetNode.neighbors[random.randint(0, len(self.targetNode.neighbors) - 1)])
        self.calcDir()
        self.active = True

    def calcDir(self):
        self.dir = self.targetNode.pos - self.pos
        absDir = abs(self.dir)
        self.dir[np.argmin(absDir)] = 0
        self.dir[np.argmax(absDir)] = np.sign(self.dir[np.argmax(absDir)])

    def centerOnNode(self, node):
        self.pos = node.pos.copy()
        self.pos -= np.array([self.size, self.size]) / 2  # Center the enemy on pos

    def kill(self):
        self.active = False

    def draw(self, screen):
        if self.active:
            screen.blit(self.sprite, self.pos)
            self.ws.draw(screen)
        else:
            if self.ws.active:
                self.ws.draw(screen)

    def reset(self):
        self.active = False
        self.ws.reset()