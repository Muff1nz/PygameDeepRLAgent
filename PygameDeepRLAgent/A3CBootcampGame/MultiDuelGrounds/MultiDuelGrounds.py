'''
Has not been update to fit the new design yet
'''

import threading
from queue import Queue

import numpy as np
import pygame
from A3CBootcampGame.MultiDuelGrounds.Enemy import EnemyHandler
from A3CBootcampGame.MultiDuelGrounds.GameHandler import GameHandler
from A3CBootcampGame.MultiDuelGrounds.Player import Player
from A3CBootcampGame.MultiDuelGrounds.world import World
from A3CBootcampGame.MultiDuelGrounds.physics import physicsHandler

WHITE = 255, 255, 255
# Class for the shooting grounds level in A3CBootCamp
class MultiDuelGrounds:
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        pygame.display.init()
        self.settings = settings
        self.screen = pygame.display.set_mode([settings.screenRes, settings.screenRes])
        self.gameScreen = pygame.Surface((settings.gameRes, settings.gameRes))

        self.runGame = True
        self.episodeInProgress = True
        self.frames = []
        self.episodeData = []

        self.gameCounter = 0
        self.gameDataQueue = gameDataQueue
        self.playerActionQueue = playerActionQueue
        self.playerAction = 0
        self.playerTimeStep = -1
        self.bootStrapCounter = 0

        self.world = World(settings)
        self.player = Player(settings, "./Assets/Player.png")
        self.enemyHandler = EnemyHandler(settings)
        self.physics = physicsHandler(self.world, self.player, self.enemyHandler.enemies, self.settings)
        self.gameHandler = GameHandler(self.player, self.enemyHandler)

    def runGameLoop(self):
        if not self.episodeInProgress:
            # send data to worker
            self.gameDataQueue.put(["EpisodeData", self.episodeData])
            self.gameDataQueue.put(["Score", self.gameHandler.playerScore])
            # reset game
            self.episodeData = []
            self.frames = []
            self.gameHandler.resetGame()
            self.playerTimeStep = -1
            self.bootStrapCounter = 0
            self.gameCounter = 0

        # Render stuff
        self.gameScreen.fill(WHITE)
        self.world.draw(self.gameScreen)
        self.enemyHandler.draw(self.gameScreen)
        self.player.draw(self.gameScreen)
        if self.settings.renderQuads:
            self.physics.quadTree.draw(self.gameScreen)
        if pygame.mouse.get_focused():
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
                self.gameDataQueue.put(["Bootstrap",
                                        bootStrapData,
                                        self.episodeData[0][0]])
                self.bootStrapCounter += 1

            # Send frame to agent
            self.gameDataQueue.put(["CurrentFrame", frame])
            self.playerAction = self.playerActionQueue.get()
            self.playerTimeStep += 1
            # Rewards default to 0, game handler will track causality and update
            self.episodeData.append([frame, self.playerAction, 0])

        # Update stuff
        self.player.update(self.playerAction, self.playerTimeStep)
        self.physics.update(self.playerTimeStep)
        self.enemyHandler.update()
        self.episodeInProgress = self.gameHandler.update(self.physics.events,
                                                         self.episodeData,
                                                         self.bootStrapCounter,
                                                         self.settings.bootStrapCutOff)

        # Check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameDataQueue.put(["Game closed!"])
        self.gameCounter += 1