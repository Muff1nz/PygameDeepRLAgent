import random
import numpy as np
from A3CBootcampGame.Actor import Actor

class FoodHandler:
    def __init__(self, settings, player):
        self.settings = settings
        self.player = player
        self.food = [Food(settings, [0, 0], "./Assets/Food.png")]
        self.foodSpawnRate = 30
        self.spawnSize = settings.screenRes
        self.rng = random.Random()

    def update(self, timeStep):
        if not self.food[0].active:
            self.spawnFood()

    def randomPos(self):
        pos = np.array([self.rng.random() * (self.spawnSize - 200) + 100,
                        self.rng.random() * (self.spawnSize - 200) + 100])
        return pos

    def spawnFood(self):
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
