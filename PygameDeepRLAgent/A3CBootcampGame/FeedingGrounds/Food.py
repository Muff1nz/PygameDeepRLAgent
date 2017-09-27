import random

import numpy as np

from A3CBootcampGame.Actor import Actor


class FoodHandler:
    def __init__(self, settings, player):
        self.settings = settings
        self.player = player
        self.food = []
        for _ in range(100): self.food.append(Food(self.settings, [0, 0], "./Assets/Food.png"))
        self.foodSpawnRate = 50
        self.spawnRange = [settings.gameRes*0.1, settings.gameRes*0.9]
        self.rng = random.Random()

    def update(self, timeStep):
        if not timeStep % self.foodSpawnRate:
            self.spawnFood()

    def randomPos(self):
        pos = np.array([self.rng.randrange(self.spawnRange[0], self.spawnRange[1]),
                        self.rng.randrange(self.spawnRange[0], self.spawnRange[1])])
        return pos

    def spawnFood(self):
        food = None
        for foodBit in self.food:
            if not foodBit.active:
                food = foodBit
                break
        if not food:
            food = self.food[0]
        while True:
            pos = self.randomPos()
            food.spawn(pos)
            if not self.boxCollision(food, self.player):
                break

    def boxCollision(self, box1, box2):
        if (box1.pos[1] + box1.size <= box2.pos[1] or
            box1.pos[1] >= box2.pos[1] + box2.size or
            box1.pos[0] + box1.size <= box2.pos[0] or
            box1.pos[0] >= box2.pos[0] + box2.size):
            return False
        return True

    def draw(self, screen):
        for foodBit in self.food:
            if foodBit.active:
                foodBit.draw(screen)

    def reset(self):
        timer = 0
        for foodBit in self.food:
            foodBit.active = False

class Food(Actor):
    def __init__(self, settings, pos, spritePath):
        super(Food, self).__init__(settings, spritePath, 0.1)
        self.type = "food"
        self.active = False
        self.pos = pos

    def spawn(self, pos):
        self.pos = pos
        self.active = True

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
