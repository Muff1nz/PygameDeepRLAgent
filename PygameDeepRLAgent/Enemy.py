import pygame
import numpy as np
import random
import WeaponSystem as ws

class Enemy:
    def __init__(self, settings, world):
        self.type = "enemy"
        self.settings = settings

        # Physical attributes
        self.pos = np.array([settings.screenWidth / 2, settings.screenHeight / 5])
        self.oldPos = self.pos.copy()
        self.speed = 10
        self.ws = ws.WeaponSystem(settings, "./Assets/Enemy.png")
        self.w = int(settings.screenWidth / 25)
        self.h = int(settings.screenHeight / 25)
        self.size = np.array([self.h, self.w])
        # needed for collision checking
        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.h]))
        self.vertices.append(np.array([self.w, 0]))
        self.vertices.append(np.array([self.w, self.h]))

        # Graphical attributes
        self.sprite = pygame.image.load("./Assets/Enemy.png")
        self.sprite = pygame.transform.scale(self.sprite, (self.w, self.h))

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
        self.pos -= self.size / 2  # Center the enemy on pos

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