import threading
from queue import Queue

import numpy as np
import pygame
from A3CBootcampGame.ShootingGrounds.Targets import TargetHandler
from A3CBootcampGame.ShootingGrounds.GameHandler import GameHandler
from A3CBootcampGame.ShootingGrounds.Player import Player
from A3CBootcampGame.ShootingGrounds.world import World
from A3CBootcampGame.ShootingGrounds.physics import physicsHandler

WHITE = 255, 255, 255
# Class for the shooting grounds level in A3CBootCamp
class ShootingGrounds:
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

        self.world = World(settings)
        self.player = Player(settings, "./Assets/Player.png")
        self.targetHandler = TargetHandler(settings, self.player)
        self.physics = physicsHandler(self.world, self.player, self.targetHandler, self.settings)
        self.gameHandler = GameHandler(self.player, self.targetHandler)

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
        self.targetHandler.draw(self.gameScreen)
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
            # Send frame to agent
            self.gameDataQueue.put(["CurrentFrame", frame])
            self.playerAction = self.playerActionQueue.get()
            self.playerTimeStep += 1
            # Rewards default to 0, game handler will track causality and update
            self.episodeData.append([frame, self.playerAction, 0])

        # Update stuff
        self.player.update(self.playerAction, self.playerTimeStep)
        self.physics.update(self.playerTimeStep)
        self.targetHandler.update(self.gameCounter)
        self.episodeInProgress = self.gameHandler.update(self.physics.events, self.gameCounter, self.episodeData)

        # Check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameDataQueue.put(["Game closed!"])
        self.gameCounter += 1