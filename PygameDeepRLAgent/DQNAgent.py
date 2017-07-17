import numpy as np
import random

from Actor import Actor

class DQNAgent(Actor):
    def __init__(self, settings, spritePath):
        super(DQNAgent, self).__init__(settings, spritePath)
        self.type = "player"
        self.lastAction = 0

    def update(self, ei):
        dir = np.array([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])
        action = random.randint(0, 24) # In the future, get action value from a CNN
        self.lastAction = action

        self.oldPos = self.pos.copy()
        self.pos += dir[action // 5] * self.speed
        d = dir[action % 5]
        if not (d[0] == 0 and d[1] == 0):
            self.ws.shoot(d, self.pos + int(self.size / 2), experienceIndex=ei)

        self.ws.update()

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
        self.ws.draw(screen)

    def reset(self):
        self.pos = np.array([self.settings.screenRes / 2, self.settings.screenRes / 5])
        self.ws.reset()

