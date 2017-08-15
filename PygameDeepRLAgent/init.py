class Settings():
    def __init__(self):
        self.version = "0.92"
        self.agentName = "A3C"
        self.logProcessedFrames = False # Doing this is very expensive

        self.screenRes = 1024 # Screen is always a square
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 6000 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame

        self.quadTreeDepth = 2
        self.quadTreeMaxObjects = 10
        self.renderQuads = False

        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent
        self.downSampleFactor = 64
        self.processedRes = self.screenRes // (self.screenRes // self.downSampleFactor)
        if self.screenRes % self.downSampleFactor: # Downsampling will make an extra sample
            self.processedRes += 1
        self.gamma = 0.99
        self.workerCount = 8
        self.maxEpisodeLength = 300
        self.bootStrapCutOff = 200

        self.tfGraphPath = 'C:/deepRLAgent/Agent/' + self.agentName + "_" + self.version
        self.tfCheckpoint = 2281716 # Check point to load, this gets set automatically when saving
        self.loadCheckpoint = True
        self.saveCheckpoint = True
        self.tbPath = 'C:/deepRLAgent/tensorboard/' + self.agentName + "_" + self.version # path for storing tensorboard logs

        self.gpuMemoryFraction = 0.66