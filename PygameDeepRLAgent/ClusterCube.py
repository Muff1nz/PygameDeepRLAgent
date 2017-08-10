import pygame
import sys
import threading
from queue import Queue
import numpy as np

from EnemyHandler import EnemyHandler
from GameHandler import GameHandler
from Player import Player
from physics import physicsHandler
from world import World

WHITE = 255, 255, 255
class ClusterCube:
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        pygame.init()
        self.settings = settings
        self.screen = pygame.display.set_mode([settings.screenRes, settings.screenRes])

        self.runGame = True
        self.episodeInProgress = True
        self.processedFrames = []
        self.processedFramesLock = threading.Lock()
        self.frameQueue = Queue(100)
        self.frameProcessingThread = threading.Thread(target=self.processFrames, name="imageProcessor",
                                                      args=(self.frameQueue, self.processedFramesLock))
        self.frameProcessingThread.start()
        self.episodeData = []

        self.gameCounter = 0

        self.gameDataQueue = gameDataQueue
        self.playerActionQueue = playerActionQueue

        self.world = World(settings)
        self.player = Player(settings, "./Assets/Player.png")
        self.playerAction = 0
        self.playerTimeStep = -1
        self.enemyHandler = EnemyHandler(settings, "./Assets/Enemy.png", self.world)
        self.physics = physicsHandler(self.world, self.player, self.enemyHandler, self.settings)
        self.gameHandler = GameHandler(self.physics.events, self.enemyHandler, self.player, self.episodeData)

    def processFrames(self, q, lock):
        while 1:
            frame = q.get()

            if frame == -1:
                q.task_done()
                return

            #down sample
            r = self.settings.screenRes // self.settings.downSampleFactor
            frame = pygame.surfarray.pixels3d(frame)[::r, ::r]

            #convert to grayscale
            grayScaleFrame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            with lock:
                self.processedFrames.append(grayScaleFrame)
                q.task_done()

    def getCurrentFrame(self):
        self.frameQueue.join()
        with self.processedFramesLock:
            return self.processedFrames[len(self.processedFrames)-1]

    def getEpisodeData(self):
        with self.processedFramesLock:
            return [self.processedFrames, self.episodeData]

    def runGameLoop(self):
        if not self.episodeInProgress:
            self.episodeData[len(self.episodeData)-1][2] = -1  # Assign reward of -1 because player lost
            self.gameDataQueue.put(["EpisodeData", self.episodeData])
            self.episodeData = []
            self.gameHandler.resetGame()
            self.playerTimeStep = -1

        # Render stuff
        self.screen.fill(WHITE)
        self.world.draw(self.screen)
        self.enemyHandler.draw(self.screen)
        self.player.draw(self.screen)
        if self.settings.renderQuads:
            self.physics.quadTree.draw(self.screen)
        pygame.display.flip()


        if not self.gameCounter % self.settings.deepRLRate:
            self.frameQueue.put(self.screen.copy())
            # Put current frame on queue, so that worker agent can compute an action
            self.gameDataQueue.put(["CurrentFrame", self.getCurrentFrame()])
            self.playerAction = self.playerActionQueue.get()
            self.playerTimeStep += 1
            # Rewards default to 0, game handler will track causality and update
            self.episodeData.append([self.getCurrentFrame(), self.playerAction, 0, None])
            if len(self.episodeData) > 1:
                self.episodeData[len(self.episodeData) - 2][3] = self.getCurrentFrame()

        # Update stuff
        self.enemyHandler.update()
        self.player.update(self.playerAction, self.playerTimeStep)
        self.physics.update(self.playerTimeStep)
        self.episodeInProgress = self.gameHandler.update(self.episodeData)

        # Check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.gameCounter += 1