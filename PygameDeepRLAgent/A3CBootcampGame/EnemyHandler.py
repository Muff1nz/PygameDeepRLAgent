import random

from ClusterCubeGame import Enemy


class EnemyHandler:
    def __init__(self, settings, spritePath, world):
        self.spritePath = spritePath
        self.enemies = []
        self.timer = 0
        self.settings = settings
        self.world = world

    def update(self):
        for enemy in self.enemies:
            enemy.update()

        self.timer += 0.1
        if random.uniform(0, 1) < 0.01:
            self.spawn()

    def spawn(self):
        for enemy in self.enemies:
            if not enemy.active:
                enemy.spawn()
                return
        self.enemies.append(Enemy.Enemy(self.settings, self.spritePath, self.world))

    def draw(self, screen):
        for enemy in self.enemies:
            enemy.draw(screen)

    def reset(self):
        self.timer = 0
        for enemy in self.enemies:
            enemy.reset()