import asyncio
import numpy as np
import pygame

from A3CBootcampGame.BaseGame import BaseGame
from A3CBootcampGame.MultiDuelGrounds.Enemy import EnemyHandler
from A3CBootcampGame.MultiDuelGrounds.GameHandler import GameHandler
from A3CBootcampGame.MultiDuelGrounds.Player import Player
from A3CBootcampGame.MultiDuelGrounds.world import World
from A3CBootcampGame.physics import physicsHandler

WHITE = 255, 255, 255
# Class for the multi duel grounds level in A3CBootCamp
class MultiDuelGrounds(BaseGame):
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        BaseGame.__init__(self, settings, gameDataQueue, playerActionQueue)

    def initGame(self):
        self.baseInit()

        self.world = World(self.settings)
        self.player = Player(self.settings, "./Assets/Player.png")
        self.enemyHandler = EnemyHandler(self.settings)
        self.gameHandler = GameHandler(self.player, self.enemyHandler)

        collisionGroups = 2
        boxes = []
        self.player.collisionGroup = 0
        boxes.append(self.player)
        for bullet in self.player.ws.bullets:
            bullet.collisionGroup = 0
            boxes.append(bullet)

        for enemy in self.enemyHandler.enemies:
            enemy.collisionGroup = 1
            boxes.append(enemy)
            for bullet in enemy.ws.bullets:
                bullet.collisionGroup = 1
                boxes.append(bullet)

        self.physics = physicsHandler(self.world.walls, boxes, collisionGroups, self.settings)

    async def run(self):
        while True:
            if not self.episodeInProgress:
                self.endEpisode()

            # Render stuff
            self.gameScreen.fill(WHITE)
            self.world.draw(self.gameScreen)
            self.enemyHandler.draw(self.gameScreen)
            self.player.draw(self.gameScreen)
            if self.window:
               self.drawWindow()

            if not self.gameCounter % self.settings.deepRLRate:
                self.sendFrameToWorker()
                while self.playerActionQueue.empty():  # Yield to let other games run, to prevent blocking on the queue
                    await asyncio.sleep(0.005)
                self.getActionFromWorker()

            # Update stuff
            self.player.update(self.playerAction, self.timeStep)
            self.physics.update(self.timeStep)
            self.enemyHandler.update()
            self.episodeInProgress = self.gameHandler.update(self.physics.events,
                                                             self.episodeData,
                                                             self.bootStrapCounter,
                                                             self.settings.bootStrapCutOff)

            if self.window:
                self.handleWindow()
            self.gameCounter += 1