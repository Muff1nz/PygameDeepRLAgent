import random
import numpy as np
from A3CBootcampGame.Actor import Actor

class FoodHandler:
    def __init__(self, settings):
        self.settings = settings
        self.food = []
        self.foodSpawnRate = 30
        self.spawnSize = settings.screenRes - 100

    def update(self, timeStep):
        if not timeStep % self.foodSpawnRate:
            self.spawnFood()

    def spawnFood(self):
        pos = np.array([random.random()*self.spawnSize,
                        random.random()*self.spawnSize])
        for foodBit in self.food:
            if not foodBit.active:
                foodBit.spawn(pos)
                return
        self.food.append(Food(self.settings, pos, "./Assets/Food.png"))

    def draw(self, screen):
        for foodBit in self.food:
            if foodBit.active:
                foodBit.draw(screen)

    def reset(self):
        for foodBit in self.food:
            foodBit.active = False

class Food(Actor):
    def __init__(self, settings, pos, spritePath):
        super(Food, self).__init__(settings, spritePath)
        self.active = True
        self.pos = pos

    def spawn(self, pos):
        self.pos = pos
        self.active = True

    def playerCollision(self):
        self.active = False

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
