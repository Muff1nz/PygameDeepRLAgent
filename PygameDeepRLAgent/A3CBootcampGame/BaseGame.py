import asyncio
import numpy as np
import pygame
from abc import abstractmethod

WHITE = 255, 255, 255
# Class for the multi duel grounds level in A3CBootCamp
class BaseGame():
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        self.settings = settings
        self.gameDataQueue = gameDataQueue
        self.playerActionQueue = playerActionQueue
        self.initGame()

    def baseInit(self):
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

    @abstractmethod
    def initGame(self):
        self.baseInit()
        pass

    def bootStrap(self):
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

    def endEpisode(self):
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

    def drawWindow(self):
        if pygame.mouse.get_focused():
            if self.settings.renderQuads:
                self.physics.quadTree.draw(self.gameScreen)
            montiroFrame = pygame.transform.scale(self.gameScreen.copy(),
                                                  (self.settings.screenRes, self.settings.screenRes))
            self.screen.blit(montiroFrame, montiroFrame.get_rect())
            pygame.display.flip()

    def handleWindow(self):
        # Check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameDataQueue.put([self.worker, ["Game closed!"]])

    def sendFrameToWorker(self, bootstrap):
        frame = pygame.surfarray.array3d(self.gameScreen.copy())
        frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

        if (bootstrap):
            self.bootStrap()

        # Send frame to agent
        self.gameDataQueue.put([self.worker, ["CurrentFrame", frame]])
        return frame


    def getActionFromWorker(self, frame):
        self.playerAction = self.playerActionQueue.get()
        self.timeStep += 1
        # Rewards default to 0, game handler will track causality and update
        self.episodeData.append([frame, self.playerAction, 0])

    @abstractmethod
    async def run(self):
        pass