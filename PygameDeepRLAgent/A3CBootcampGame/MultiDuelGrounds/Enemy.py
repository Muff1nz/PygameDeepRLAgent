import numpy as np
import random

from A3CBootcampGame.Actor import Actor
from A3CBootcampGame.WeaponSystem import WeaponSystem

class EnemyHandler():
    def __init__(self, settings):
        res = settings.gameRes
        wallDist = res/8
        self.enemies = []
        self.enemyRespawnChance = 0.005

        spawnGen = lambda: np.array([random.randint(res/4, res - res/4), wallDist])
        enemy = Enemy(settings, "./Assets/Enemy.png")
        enemy.setup(np.array([1, 0]), np.array([0, 1]), spawnGen)
        self.enemies.append(enemy)

        spawnGen = lambda: np.array([random.randint(res/4, res - res/4), res - wallDist])
        enemy = Enemy(settings, "./Assets/Enemy.png")
        enemy.setup(np.array([1, 0]), np.array([0, -1]), spawnGen)
        self.enemies.append(enemy)

        spawnGen = lambda: np.array([wallDist, random.randint(res / 4, res - res / 4)])
        enemy = Enemy(settings, "./Assets/Enemy.png")
        enemy.setup(np.array([0, 1]), np.array([1, 0]), spawnGen)
        self.enemies.append(enemy)

        spawnGen = lambda: np.array([res - wallDist, random.randint(res / 4, res - res / 4)])
        enemy = Enemy(settings, "./Assets/Enemy.png")
        enemy.setup(np.array([0, 1]), np.array([-1, 0]), spawnGen)
        self.enemies.append(enemy)

    def update(self):
        for enemy in self.enemies:
            if enemy.active:
                enemy.update()
            else:
                if self.enemyRespawnChance > random.random():
                    enemy.spawn()

    def draw(self, screen):
        for enemy in self.enemies:
            if enemy.active:
                enemy.draw(screen)

    def reset(self):
        for enemy in self.enemies:
            enemy.reset()

class Enemy(Actor):
    def __init__(self, settings, spritePath):
        super(Enemy, self).__init__(settings, spritePath, 0.075)
        self.type = "enemy"
        self.speed = settings.gameRes * 0.01
        self.ws = WeaponSystem(settings, spritePath, 0.075 / 2)
        self.strafeDir = np.zeros(2)
        self.shootDir = np.zeros(2)

        self.spawnGen = None
        self.active = False

    def setup(self, strafeDir, shootDir, spawnGen):
        self.strafeDir = strafeDir
        self.shootDir = shootDir
        self.spawnGen = spawnGen

    def spawn(self):
        self.active = True
        self.pos = self.spawnGen()

    def update(self):
        self.ws.update()
        self.oldPos = self.pos.copy()
        self.pos += self.strafeDir * self.speed
        self.ws.shoot(self.shootDir, self.pos + self.size/2)

    def onWallCollision(self):
        self.strafeDir *= -1

    def onBoxCollision(self, other):
        if (other.type == "bullet"):
            self.active = False

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
        self.ws.draw(screen)

    def reset(self):
        self.active = False
        self.ws.reset()
