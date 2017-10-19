import asyncio
import numpy as np
import pygame

from A3CBootcampGame.MultiDuelGrounds.Enemy import EnemyHandler
from A3CBootcampGame.MultiDuelGrounds.GameHandler import GameHandler
from A3CBootcampGame.MultiDuelGrounds.Player import Player
from A3CBootcampGame.MultiDuelGrounds.world import World
from A3CBootcampGame.physics import physicsHandler

WHITE = 255, 255, 255
# Class for the multi duel grounds level in A3CBootCamp
class MultiDuelGrounds():
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        self.settings = settings
        self.gameDataQueue = gameDataQueue
        self.playerActionQueue = playerActionQueue
        self.initGame()

    def initGame(self):
        initSettings = self.playerActionQueue.get()
        self.window = initSettings["WindowSettings"]
        self.worker = initSettings["worker"]

        if self.window:
            pygame.display.init()
            self.screen = pygame.display.set_mode([self.settings.screenRes, self.settings.screenRes])
        self.gameScreen = pygame.Surface((self.settings.gameRes, self.settings.gameRes))

        self.runGame = True
        self.episodeInProgress = True
        self.frames = []
        self.episodeData = []

        self.gameCounter = 0

        self.playerAction = 0
        self.timeStep = -1
        self.bootStrapCounter = 0

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
                # send data to worker
                self.gameDataQueue.put([self.worker,
                                       ["EpisodeData", np.array(self.episodeData), self.gameHandler.playerScore]])
                # reset game
                self.episodeData = []
                self.frames = []
                self.gameHandler.resetGame()
                self.timeStep = -1
                self.bootStrapCounter = 0
                self.gameCounter = 0

            # Render stuff
            self.gameScreen.fill(WHITE)
            self.world.draw(self.gameScreen)
            self.enemyHandler.draw(self.gameScreen)
            self.player.draw(self.gameScreen)
            if self.window:
                if pygame.mouse.get_focused():
                    if self.settings.renderQuads:
                        self.physics.quadTree.draw(self.gameScreen)
                    montiroFrame = pygame.transform.scale(self.gameScreen.copy(),
                                                         (self.settings.screenRes, self.settings.screenRes))
                    self.screen.blit(montiroFrame, montiroFrame.get_rect())
                    pygame.display.flip()

            if not self.gameCounter % self.settings.deepRLRate:
                frame = pygame.surfarray.array3d(self.gameScreen.copy())
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

                # Send the already reward credited data + an extra frame to the worker,
                # so that he can use it to compute the value of the state and
                # Bootstrap the learning from the current knowledge.
                if self.settings.maxEpisodeLength <= len(self.episodeData):
                    # Due to causality tracking, we cant bootstrap from recent frames,
                    # as reward crediting is still in progress
                    bootStrapData = self.episodeData[0:self.settings.bootStrapCutOff]
                    self.episodeData = self.episodeData[self.settings.bootStrapCutOff::]
                    self.gameDataQueue.put([self.worker,
                                            ["Bootstrap",
                                            bootStrapData,
                                            self.episodeData[0][0]]])
                    self.bootStrapCounter += 1

                # Send frame to agent
                self.gameDataQueue.put([self.worker, ["CurrentFrame", frame]])
                while self.playerActionQueue.empty():  # Yield to let other games run, to prevent blocking on the queue
                    await asyncio.sleep(0.005)
                self.playerAction = self.playerActionQueue.get()
                self.timeStep += 1
                # Rewards default to 0, game handler will track causality and update
                self.episodeData.append([frame, self.playerAction, 0])

            # Update stuff
            self.player.update(self.playerAction, self.timeStep)
            self.physics.update(self.timeStep)
            self.enemyHandler.update()
            self.episodeInProgress = self.gameHandler.update(self.physics.events,
                                                             self.episodeData,
                                                             self.bootStrapCounter,
                                                             self.settings.bootStrapCutOff)

            if self.window:
                # Check events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.gameDataQueue.put([self.worker, ["Game closed!"]])
            self.gameCounter += 1