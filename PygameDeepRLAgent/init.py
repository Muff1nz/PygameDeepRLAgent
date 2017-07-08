class Settings():
    def __init__(self):
        self.screenRes = 1000 # Screen is always a square
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 60 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame

        self.experienceMemorySize = 100000 # Length of expeirence memory
        self.deepRLSampleRate = 8 # how many frames to wait for sampling experiences for deepRLAgent
        self.downSampleFactor = 64
        self.processedRes = self.screenRes // (self.screenRes // self.downSampleFactor)
        if self.screenRes % self.downSampleFactor: # Downsampling will make an extra sample
            self.processedRes += 1

        self.version = 0.1
        self.tbPath = "/deepRLAgent/tensorboard/" + str(self.version) # path for storing tensorboard logs