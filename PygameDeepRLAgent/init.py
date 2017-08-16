class Settings():
    def __init__(self):
        # General settings:
        self.version = "0.95_OvernightTest"
        self.agentName = "A3C"
        self.gpuMemoryFraction = 0.8

        # Game settings:
        self.screenRes = 1024 # Screen is always a square
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 6000 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame
        self.sleepTime = 0.01 # To ease the amount of context switching if worker count > cpu count

        self.quadTreeDepth = 2
        self.quadTreeMaxObjects = 10
        self.renderQuads = False

        # If enabled, the game will assign rewards to the state-action-reward tuple that caused the reward.
        # If a bullet is shot at time step 10, and it hits a enemy at time step 15, the time step when the bullet
        # was shot will receive the reward, because that's the time step when the action causing the reward happened.
        self.causalityTracking = False

        # AI settings:
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent
        self.downSampleFactor = 64
        self.processedRes = self.screenRes // (self.screenRes // self.downSampleFactor)
        if self.screenRes % self.downSampleFactor: # Downsampling will make an extra sample
            self.processedRes += 1
        self.gamma = 0.99
        self.workerCount = 16
        self.maxEpisodeLength = 300
        self.bootStrapCutOff = 200

        self.tfGraphPath = 'C:/deepRLAgent/Agent/' + self.agentName + "_" + self.version
        self.tfCheckpoint = 2281716 # Check point to load, this gets set automatically when saving
        self.loadCheckpoint = True
        self.saveCheckpoint = True
        self.tbPath = 'C:/deepRLAgent/tensorboard/' + self.agentName + "_" + self.version
        self.gifPath = 'C:/deepRLAgent/gif/' + self.agentName + "_" + self.version