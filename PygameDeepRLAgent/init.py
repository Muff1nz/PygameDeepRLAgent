class Settings():
    def __init__(self):
        # General settings:
        self.version = "0.99"
        self.agentName = "A3C"
        self.activity = "shootingGroundsCTFalse"
        self.gpuMemoryFraction = 1.00

        # Game settings:
        self.screenRes = 1000 # Screen is always a square
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 60 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame
        self.sleepTime = 0.0016 # To ease the amount of context switching if worker count > cpu count

        self.quadTreeDepth = 2
        self.quadTreeMaxObjects = 10
        self.renderQuads = False

        # If enabled, the game will assign rewards to the state-action-reward tuple that caused the reward.
        # If a bullet is shot at time step 10, and it hits a enemy at time step 15, the time step when the bullet
        # was shot will receive the reward, because that's the time step when the action causing the reward happened.
        self.causalityTracking = False

        # AI settings:
        self.trainingEpisodes = 10000

        # Hyper parameters:
        self.gameRes = 80
        self.actionSize = 9
        self.gamma = 0.99
        self.workerCount = 16
        self.maxEpisodeLength = 300
        self.bootStrapCutOff = 200
        self.learningRate = 5e-5
        self.lrDecayRate = 0.95
        self.lrDecayStep = 200
        self.entropyWeight = 0.01
        self.valueWeight = 0.5
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent

        self.loadCheckpoint = True
        self.saveCheckpoint = True
        self.logSummaries = True
        self.train = True
        self.tfGraphPath = 'C:/deepRLAgent/Agent/' + self.activity + "_" + self.agentName + "_" + self.version
        self.tfCheckpoint = 'C:/deepRLAgent/Agent/shootingGroundsCTFalse_A3C_0.99A3C-6001'  # Check point to load
        self.tbPath = 'C:/deepRLAgent/tensorboard/' + self.activity + "/" + self.agentName + "_" + self.version
        self.gifPath = 'C:/deepRLAgent/gif/' + self.agentName + "_" + self.version

    def loadFromFile(self, fileName):
        with open(fileName, 'r') as file:
            pass
