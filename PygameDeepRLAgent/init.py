import multiprocessing

from A3CBootcampGame.FeedingGrounds.FeedingGrounds import FeedingGrounds
from A3CBootcampGame.ShootingGrounds.ShootingGrounds import ShootingGrounds
from A3CBootcampGame.MultiDuelGrounds.MultiDuelGrounds import MultiDuelGrounds

class Settings():
    def __init__(self, game=None):
        # Game settings:
        if game:
            self.game = game
        else:
            self.game = "ShootingGrounds"
        self.screenRes = 1000  # Screen is always a square
        self.gameSecond = 60  # Amount of frames considered a second in game
        self.fps = 60  # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000  # miliseconds per frame

        self.quadTreeDepth = 3
        self.quadTreeMaxObjects = 6
        self.renderQuads = False

        self.games = {"FeedingGrounds": [FeedingGrounds, 4],
                      "ShootingGrounds": [ShootingGrounds, 9],
                      "MultiDuelGrounds": [MultiDuelGrounds, 9]}

        # If enabled, the game will assign rewards to the state-action-reward tuple that caused the reward.
        # If a bullet is shot at time step 10, and it hits a enemy at time step 15, the time step when the bullet
        # was shot will receive the reward, because that's the time step when the action causing the reward happened.
        self.causalityTracking = False

        # AI settings:
        self.trainingEpisodes = 30000
        self.logFreq = 10 # Log summaries every 50 episodes

        # Hyper parameters:
        self.gameRes = 80
        self.actionSize = self.games[self.game][1]
        self.gamma = 0.99
        self.trainerCount = 16
        self.workersPerTrainer = 4
        self.maxEpisodeLength = 1200 # Does not effect fixed episode length games (Feeding/Shooting grounds)
        self.bootStrapCutOff = 100
        self.learningRate = 1e-3
        self.lrDecayRate = 0.97
        self.lrDecayStep = 150
        self.entropyWeight = 0.01
        self.valueWeight = 0.5
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent

        self.loadCheckpoint = False
        self.saveCheckpoint = False
        self.logSummaries = True
        self.train = True

        # General settings:
        self.version = "1.30"
        self.generateActivity()
        self.gpuMemoryFraction = 1.0

        # File paths
        self.tfCheckpoint = 'C:/deepRLAgent/Agent/MultiDuelGrounds_ACNetwork_MDG_LSTM_5e-4LR_16T_2W_A3CMaster_1.20A3CMaster-184229'  # Check point to load
        self.generatePaths()

        # Not counting main thread, because its mostly blocked
        if (self.trainerCount * 2 > multiprocessing.cpu_count()):
            raise RuntimeWarning("The programs thread+process count is larger then system thread count, may impair performance")

    def generateActivity(self):
        self.activity = "{}LR_{}LRDR_{}LRDS_{}DLRRate_{}T-{}W_{}Episodes".format(
            self.learningRate, self.lrDecayRate, self.lrDecayStep, self.deepRLRate,
            self.trainerCount, self.workersPerTrainer, self.trainingEpisodes
        )

    def generatePaths(self):
        self.tfGraphPath = 'C:/deepRLAgent/Agent/{}_{}_{}'.format(self.game, self.activity, self.version)
        self.tbPath = 'C:/deepRLAgent/tensorboard/{}/{}/{}'.format(self.game, self.activity, self.version)
