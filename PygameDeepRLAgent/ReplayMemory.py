import pygame
import numpy as np
import threading
from queue import Queue

# Experience tuple
class Experience:
    def __init__(self, settings):
        self.state = np.ndarray((settings.stateDepth, settings.processedRes, settings.processedRes), dtype='float32')
        self.action = 0
        self.reward = 0
        self.newState = np.ndarray((settings.stateDepth, settings.processedRes, settings.processedRes), dtype='float32')


class ReplayMemory:
    def __init__(self, settings):
        self.settings = settings
        self.ei = -1 # Experience index
        # Experience memory entires consist of (action, reward, experience index)
        self.experienceMemory = np.ndarray(shape=(settings.experienceMemorySize, 3), dtype='int32') #Contains experience index, action and reward, states will be constructed from ei
        self.processedFrames = np.ndarray(
            shape=(settings.experienceMemorySize, settings.processedRes, settings.processedRes),
            dtype='float32'
        )
        self.processedFramesLock = threading.Lock()
        self.frameQueue = Queue(100)
        self.frameProcessingThread = threading.Thread(target=self.processFrames, name="imageProcessor", args=(self.frameQueue, self.processedFramesLock))
        self.frameProcessingThread.start()
        self.full = False

    def update(self, screen, player):
        self.ei = (self.ei + 1) % self.settings.experienceMemorySize

        if self.ei == self.settings.experienceMemorySize and not self.full:
            self.full = True

        self.experienceMemory[self.ei, 0] = player.lastAction
        self.experienceMemory[self.ei, 1] = 0 # Will be set by gamehandler
        self.experienceMemory[self.ei, 2] = self.ei
        self.frameQueue.put([screen.copy(), self.ei])

    #This method runs in its own thread, processing frames as they become available
    def processFrames(self, q, lock):
        while 1:
            item = q.get()
            frame = item[0]
            ei = item[1]

            if ei == -1:
                return

            #down sample
            r = self.settings.screenRes // self.settings.downSampleFactor
            frame = pygame.surfarray.pixels3d(frame)[::r, ::r]

            #convert to grayscale
            grayScaleFrame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            with lock:
                self.processedFrames[ei] = grayScaleFrame
                q.task_done()

    def getState(self, ei=-1):
        if ei == -1:
            ei = self.ei

        state = np.ndarray(
            shape=[self.settings.stateDepth, self.settings.processedRes, self.settings.processedRes],
            dtype='float32'
        )

        self.frameQueue.join()
        with self.processedFramesLock:
            for i in range(self.settings.stateDepth):
                state[i] = self.processedFrames[(ei - i) % self.settings.experienceMemorySize].copy()
        return state

    #Might want to change this to use tensorflow queues instead of python dicts
    def getExperienceBatch(self, size):
        batch = [Experience(self.settings)] * size
        if self.full:
            item = self.experienceMemory[np.random.randint(0, self.settings.experienceMemorySize)]
        else:
            item = self.experienceMemory[np.random.randint(self.settings.stateDepth, self.ei)]
        for experience in batch:
            experience.action = item[0]
            experience.reward = item[1]
            experience.state = self.getState(item[2])
            experience.newState = self.getState(item[2] + 1)

        return batch

    def close(self):
        self.frameQueue.put([-1, -1])
