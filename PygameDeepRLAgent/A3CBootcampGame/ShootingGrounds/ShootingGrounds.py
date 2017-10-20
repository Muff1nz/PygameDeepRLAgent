import asyncio
import numpy as np
import pygame

from A3CBootcampGame.BaseGame import BaseGame
from A3CBootcampGame.ShootingGrounds.Targets import TargetHandler
from A3CBootcampGame.ShootingGrounds.GameHandler import GameHandler
from A3CBootcampGame.ShootingGrounds.Player import Player
from A3CBootcampGame.ShootingGrounds.world import World
from A3CBootcampGame.physics import physicsHandler

WHITE = 255, 255, 255
# Class for the shooting grounds level in A3CBootCamp
class ShootingGrounds(BaseGame):
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        BaseGame.__init__(self, settings, gameDataQueue, playerActionQueue)

    def initGame(self): # This function was created to do init in the run function when this class was a process
        self.baseInit()

        self.world = World(self.settings)
        self.player = Player(self.settings, "./Assets/Player.png")
        self.targetHandler = TargetHandler(self.settings, self.player)
        self.gameHandler = GameHandler(self.player, self.targetHandler)

        collisionGroups = 2
        boxes = []
        self.player.collisionGroup = 0
        boxes.append(self.player)
        for bullet in self.player.ws.bullets:
            bullet.collisionGroup = 0
            boxes.append(bullet)
        self.targetHandler.target.collisionGroup = 1
        boxes.append(self.targetHandler.target)
        self.physics = physicsHandler(self.world.walls, boxes, collisionGroups, self.settings)

    async def run(self):
        while True:
            if not self.episodeInProgress:
                self.endEpisode()

            # Render stuff
            self.gameScreen.fill(WHITE)
            self.world.draw(self.gameScreen)
            self.targetHandler.draw(self.gameScreen)
            self.player.draw(self.gameScreen)
            if self.window:
                self.drawWindow()

            if not self.gameCounter % self.settings.deepRLRate:
                frame = self.sendFrameToWorker(bootstrap=False)
                while self.playerActionQueue.empty():  # Yield to let other games run, to prevent blocking on the queue
                    await asyncio.sleep(0.005)
                self.getActionFromWorker(frame)

            # Update stuff
            self.player.update(self.playerAction, self.timeStep)
            self.physics.update(self.timeStep)
            self.targetHandler.update(self.gameCounter)
            self.episodeInProgress = self.gameHandler.update(self.physics.events, self.gameCounter, self.episodeData)

            if self.window:
                self.handleWindow()
            self.gameCounter += 1