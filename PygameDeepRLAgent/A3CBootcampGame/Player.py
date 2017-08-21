import numpy as np
from A3CBootcampGame.Actor import Actor

class Player(Actor):
    def __init__(self, settings, spritePath):
        super(Player, self).__init__(settings, spritePath)
        self.type = "player"

    def update(self, action, plc):
        dir = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])

        self.oldPos = self.pos.copy()
        self.pos += dir[action] * self.speed

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)

    def reset(self):
        self.pos = np.array([self.settings.screenRes / 2, self.settings.screenRes / 5])
















