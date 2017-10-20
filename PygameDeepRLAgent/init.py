import multiprocessing
import os
import warnings

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
        self.trainingEpisodes = 1000000
        self.logFreq = 10 # Log summaries every 50 episodes

        # Training config
        self.trainerThreads = 8
        self.workerThreads = 8
        self.gameProcesses = 16

        self.trainers = 32
        self.workers = 32

        assert(self.trainers % self.trainerThreads == 0 and self.trainers > 0)
        assert(self.workers % self.workerThreads == 0 and self.workers > 0)
        assert(self.workers % self.gameProcesses == 0)
        assert(self.workers % self.trainers == 0)

        # Hyper parameters:
        self.gameRes = 80
        self.actionSize = self.games[self.game][1]
        self.gamma = 0.99
        self.maxEpisodeLength = 1200 # Does not effect fixed episode length games (Feeding/Shooting grounds)
        self.bootStrapCutOff = 100
        self.learningRate = 5e-5
        self.lrDecayRate = 0.97
        self.lrDecayStep = 150
        self.entropyWeight = 0.01
        self.valueWeight = 0.5
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent
        self.frameSequenceLen = 3 # How many frames to stack together

        self.loadCheckpoint = False
        self.saveCheckpoint = True
        self.logSummaries = True
        self.train = True

        # General settings:
        self.version = "1.37"
        self.generateActivity()

        # File paths
        self.tfCheckpoint = 'not set'  # Check point to load
        self.generatePaths()

        # Not counting main thread, because its mostly blocked
        if (self.trainerThreads + self.workerThreads + self.gameProcesses > multiprocessing.cpu_count()):
            warnings.warn("The programs thread+process count is larger then system thread count, may impair performance")

    def generateActivity(self):
        self.activity = "{}LR_{}LRDR_{}LRDS_{}DLRRate_{}T-{}W_{}Episodes".format(
            self.learningRate, self.lrDecayRate, self.lrDecayStep, self.deepRLRate,
            self.trainers, self.workers, self.trainingEpisodes
        )

    def generatePaths(self):
        self.tfGraphPath = 'C:/deepRLAgent/Agent/{}/{}/'.format(self.version, self.activity)
        self.tbPath = 'C:/deepRLAgent/tensorboard/{}/{}/{}'.format(self.version, self.game, self.activity)
        self.imagePath = 'C:/deepRLAgent/workerFrames/{}/{}/{}'.format(self.version, self.game, self.activity)

        if not os.path.exists(self.tfGraphPath):
            os.makedirs(self.tfGraphPath)
            print("Path \"{}\" did not exist, so i made it".format(self.tfGraphPath))

