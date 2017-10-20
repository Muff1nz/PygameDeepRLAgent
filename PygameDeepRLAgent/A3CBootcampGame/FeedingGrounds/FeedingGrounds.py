import asyncio
import numpy as np
import pygame

from A3CBootcampGame.BaseGame import BaseGame
from A3CBootcampGame.FeedingGrounds.Food import FoodHandler
from A3CBootcampGame.FeedingGrounds.GameHandler import GameHandler
from A3CBootcampGame.FeedingGrounds.Player import Player
from A3CBootcampGame.FeedingGrounds.world import World
from A3CBootcampGame.physics import physicsHandler

WHITE = 255, 255, 255
# Class for the feeding grounds level in A3CBootCamp
class FeedingGrounds(BaseGame):
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        BaseGame.__init__(self, settings, gameDataQueue, playerActionQueue)

    def initGame(self):
        self.baseInit()

        self.world = World(self.settings)
        self.player = Player(self.settings, "./Assets/Player.png")
        self.foodHandler = FoodHandler(self.settings, self.player)
        self.gameHandler = GameHandler(self.player, self.foodHandler)

        collisionGroups = 2
        boxes = []
        self.player.collisionGroup = 0
        boxes.append(self.player)
        for foodBit in self.foodHandler.food:
            foodBit.collisionGroup = 1
            boxes.append(foodBit)

        self.physics = physicsHandler(self.world.walls, boxes,  collisionGroups, self.settings)

    async def run(self):
        while True:
            if not self.episodeInProgress:
                self.endEpisode()

            # Render stuff
            self.gameScreen.fill(WHITE)
            self.world.draw(self.gameScreen)
            self.foodHandler.draw(self.gameScreen)
            self.player.draw(self.gameScreen)
            if self.window:
                self.drawWindow()

            if not self.gameCounter % self.settings.deepRLRate:
                frame = self.sendFrameToWorker(bootstrap=False)
                while self.playerActionQueue.empty():  # Yield to let other games run, to prevent blocking on the queue
                    await asyncio.sleep(0.005)
                self.getActionFromWorker(frame)

            # Update stuff
            self.player.update(self.playerAction)
            self.physics.update(self.timeStep)
            self.foodHandler.update(self.gameCounter)
            self.episodeInProgress = self.gameHandler.update(self.physics.events, self.gameCounter, self.episodeData)

            if self.window:
                if self.window:
                    self.handleWindow()
            self.gameCounter += 1