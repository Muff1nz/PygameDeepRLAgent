class Settings():
    def __init__(self):
        self.version = "0.9"
        self.agentName = "dqn_1"
        self.logProcessedFrames = False # Doing this is very expensive

        self.screenRes = 1024 # Screen is always a square
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 6000 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame

        self.quadTreeDepth = 2
        self.quadTreeMaxObjects = 10
        self.renderQuads = False

        self.experienceMemorySize = 100000 # Length of expeirence memory
        self.experienceMemorySizeStart = 1000
        self.experienceBatchSize = 32
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent
        self.downSampleFactor = 64
        self.processedRes = self.screenRes // (self.screenRes // self.downSampleFactor)
        if self.screenRes % self.downSampleFactor: # Downsampling will make an extra sample
            self.processedRes += 1
        self.stateDepth = 4

        self.replayMemoryPath = 'C:/deepRLAgent/Memory/' + self.agentName + "_" + self.version
        self.tfGraphPath = 'C:/deepRLAgent/Agent/' + self.agentName + "_" + self.version
        self.tfCheckpoint = 18886 # Check point to load, this gets set automatically when saving
        self.loadCheckpoint = True
        self.saveCheckpoint = True
        self.tbPath = 'C:/deepRLAgent/tensorboard/' + self.agentName + "_" + self.version # path for storing tensorboard logs

        self.gpuMemoryFraction = 0.66