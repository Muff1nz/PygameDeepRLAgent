import numpy as np

from A3CBootcampGame.Actor import Actor
from A3CBootcampGame.WeaponSystem import WeaponSystem


class Player(Actor):
    def __init__(self, settings, spritePath):
        super(Player, self).__init__(settings, spritePath, 0.1)
        self.type = "player"
        self.speed = settings.gameRes * 0.01
        self.ws = WeaponSystem(settings, spritePath)
        self.moveDir = np.array([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])
        self.shootDir = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def update(self, action, pts):
        self.ws.update()
        if action <= 4:
            self.oldPos = self.pos.copy()
            self.pos += self.moveDir[action] * self.speed
        else:
            action = action - 5
            self.ws.shoot(self.shootDir[action], self.pos + self.size/2, pts)

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
        self.ws.draw(screen)

    def reset(self):
        self.pos = np.array([self.settings.gameRes / 2, self.settings.gameRes / 2])
















