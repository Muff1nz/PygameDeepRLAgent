import numpy as np

from A3CBootcampGame.Actor import Actor


class Player(Actor):
    def __init__(self, settings, spritePath):
        super(Player, self).__init__(settings, spritePath, 0.1)
        self.type = "player"
        self.speed = settings.gameRes * 0.01

    def update(self, action):
        dir = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])

        self.oldPos = self.pos.copy()
        self.pos += dir[action] * self.speed

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)

    def reset(self):
        self.pos = np.array([self.settings.gameRes / 2, self.settings.gameRes / 2])
















