<<<<<<< HEAD
class Settings():
    def __init__(self):
        # General settings:
        self.version = "1.01"
        self.agentName = "Bobby"
        self.activity = "DistributedTF"
        self.gpuMemoryFraction = 0.20
=======
import multiprocessing
import os
import warnings
>>>>>>> origin/master

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
        self.trainingEpisodes = 100000
        self.logFreq = 10 # Log summaries every 50 episodes

        # Hyper parameters:
        self.gameRes = 80
        self.actionSize = self.games[self.game][1]
        self.gamma = 0.99
<<<<<<< HEAD
        self.trainerCount = 1
        self.workersPerTrainer = 1
        self.psCount = 1
        self.maxEpisodeLength = 1200
=======
        self.trainerCount = 16
        self.workersPerTrainer = 2
        self.maxEpisodeLength = 1200 # Does not effect fixed episode length games (Feeding/Shooting grounds)
>>>>>>> origin/master
        self.bootStrapCutOff = 100
        self.learningRate = 5e-5
        self.lrDecayRate = 0.98
        self.lrDecayStep = 140
        self.entropyWeight = 0.01
        self.valueWeight = 0.5
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent

        self.loadCheckpoint = True
        self.saveCheckpoint = True
        self.logSummaries = True
        self.train = True

        # General settings:
        self.version = "1.36"
        self.generateActivity()
        self.gpuMemoryFraction = 1.0

        # File paths
        self.tfCheckpoint = 'not set'  # Check point to load
        self.generatePaths()

        # Not counting main thread, because its mostly blocked
        if (self.trainerCount * 2 > multiprocessing.cpu_count()):
            warnings.warn("The programs thread+process count is larger then system thread count, may impair performance")

    def generateActivity(self):
        self.activity = "{}LR_{}LRDR_{}LRDS_{}DLRRate_{}T-{}W_{}Episodes".format(
            self.learningRate, self.lrDecayRate, self.lrDecayStep, self.deepRLRate,
            self.trainerCount, self.workersPerTrainer, self.trainingEpisodes
        )

    def generatePaths(self):
        self.tfGraphPath = 'C:/deepRLAgent/Agent/{}/'.format(self.activity)
        self.tbPath = 'C:/deepRLAgent/tensorboard/{}/{}/{}'.format(self.game, self.activity, self.version)

        if not os.path.exists(self.tfGraphPath):
            os.makedirs(self.tfGraphPath)
            print("Path \"{}\" did not exist, so i made it".format(self.tfGraphPath))

