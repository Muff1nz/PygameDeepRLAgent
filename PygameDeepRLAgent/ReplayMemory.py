import pygame
import numpy as np

class ReplayMemory:
    def __init__(self, settings):
        self.settings = settings
        self.ei = 0 # Experience index
        self.oldEi = 0
        # Experience memory entires consist of (action, reward, experience index)
        self.experienceMemory = np.ndarray(shape=(settings.experienceMemorySize, 3), dtype='int32') #Contains experience index, action and reward, states will be constructed from ei
        self.processedFrames = np.ndarray(
            shape=(settings.experienceMemorySize, settings.processedRes, settings.processedRes),
            dtype='float32'
        )

    def update(self, screen, player):
        self.oldEi = self.ei
        self.ei = (self.ei + 1) % self.settings.experienceMemorySize
        self.updateFrames(screen)
        self.experienceMemory[self.ei, 0] = player.lastAction
        self.experienceMemory[self.ei, 1] = 0 # Will be set later
        self.experienceMemory[self.ei, 2] = self.ei


    def updateFrames(self, screen):
        frame = screen.copy()

        #down sample
        r = self.settings.screenRes // self.settings.downSampleFactor
        frame = pygame.surfarray.pixels3d(frame)[::r, ::r]

        #convert to grayscale
        grayScaleFrame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        self.processedFrames[self.ei] = grayScaleFrame