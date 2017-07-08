import pygame
import numpy as np

class ReplayMemory:
    def __init__(self, settings):
        self.settings = settings
        self.ei = -1 # Experience index
        self.experienceMemory = np.ndarray(shape=(settings.experienceMemorySize, 3), dtype='int32') #Contains experience index, action and reward, states will be constructed from ei
        self.processedFrames = np.ndarray(
            shape=(settings.experienceMemorySize, settings.processedRes, settings.processedRes),
            dtype='float32')

    def update(self, screen):
        self.ei = (self.ei + 1) % self.settings.experienceMemorySize
        self.updateFrames(screen)


    def updateFrames(self, screen):
        frame = screen.copy()

        #down sample
        r = self.settings.screenRes // self.settings.downSampleFactor
        frame = pygame.surfarray.pixels3d(frame)[::r, ::r]

        #convert to grayscale
        grayScaleFrame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        self.processedFrames[self.ei] = grayScaleFrame