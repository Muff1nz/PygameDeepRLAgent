import random

import numpy as np

from A3CBootcampGame.Actor import Actor


class TargetHandler:
    def __init__(self, settings, player):
        self.settings = settings
        self.player = player
        self.target = Target(settings, [0,0], "./Assets/Enemy.png")
        self.foodSpawnRate = 30
        self.spawnRange = [settings.gameRes*0.1, settings.gameRes*0.9]
        self.rng = random.Random()
        self.spawnTarget()

    def update(self, timeStep):
        if not self.target.active:
            self.spawnTarget()

    def randomPos(self):
        pos = np.array([self.rng.randrange(self.spawnRange[0], self.spawnRange[1]),
                        self.rng.randrange(self.spawnRange[0], self.spawnRange[1])])
        return pos

    def spawnTarget(self):
        while True:
            pos = self.randomPos()
            self.target.spawn(pos)
            if not self.boxCollision(self.target, self.player):
                break

    def boxCollision(self, box1, box2):
        if (box1.pos[1] + box1.size <= box2.pos[1] or
            box1.pos[1] >= box2.pos[1] + box2.size or
            box1.pos[0] + box1.size <= box2.pos[0] or
            box1.pos[0] >= box2.pos[0] + box2.size):
            return False
        return True

    def draw(self, screen):
        self.target.draw(screen)

    def reset(self):
        self.target.active = False

class Target(Actor):
    def __init__(self, settings, pos, spritePath):
        super(Target, self).__init__(settings, spritePath, 0.1)
        self.type = "target"
        self.active = True
        self.pos = pos

    def spawn(self, pos):
        self.pos = pos
        self.active = True

    def playerBulletCollision(self):
        self.active = False

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
