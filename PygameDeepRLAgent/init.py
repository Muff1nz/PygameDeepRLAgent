class Settings():
    def __init__(self):
        self.version = "0.85"
        self.logProcessedFrames = False # Doing this is very expensive

        self.screenRes = 1024 # Screen is always a square
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 6000 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame

        self.quadTreeDepth = 2
        self.quadTreeMaxObjects = 10
        self.renderQuads = True

        self.experienceMemorySize = 100000 # Length of expeirence memory
        self.experienceMemorySizeStart = 1000
        self.experienceBatchSize = 32
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent
        self.downSampleFactor = 64
        self.processedRes = self.screenRes // (self.screenRes // self.downSampleFactor)
        if self.screenRes % self.downSampleFactor: # Downsampling will make an extra sample
            self.processedRes += 1
        self.stateDepth = 4

        self.tfGraphPath = "/deepRLAgent/DQN/dqnAgent" + self.version
        self.tfGraphCheckpoint = -1
        self.loadCheckpoint = False
        self.tbPath = "/deepRLAgent/tensorboard/" + self.version # path for storing tensorboard logs

        self.gpuMemoryFraction = 0.66