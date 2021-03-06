import numpy as np

from A3CBootcampGame.Actor import Actor

class Bullet(Actor):
    def __init__(self, settings, sprite, size=0.05):
        super(Bullet, self).__init__(settings, sprite, size)
        self.type = "bullet"
        self.active = False

        self.dir = np.array([0, 0])
        self.speed = settings.gameRes * 0.02
        self.timer = 0
        self.TTL = 3 #time to live in seconds


    def draw(self, screen):
        screen.blit(self.sprite, self.pos)

    def shoot(self, pos, dir, timeStep):
        self.active = True
        self.pos = np.array(pos)- np.array([self.size/2, self.size/2])
        self.dir = np.array(dir)
        self.timer = 0
        self.playerTimeStep = timeStep

    def update(self):
        if (self.active):
            self.timer += 1
            self.oldPos = self.pos.copy()
            self.pos += self.dir * self.speed
            if self.timer >= self.TTL * self.settings.gameSecond:
                self.active = False

    def onWallCollision(self):
        self.active = False
