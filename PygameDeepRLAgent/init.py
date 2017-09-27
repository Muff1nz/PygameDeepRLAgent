from A3CBootcampGame.FeedingGrounds.FeedingGrounds import FeedingGrounds
from A3CBootcampGame.ShootingGrounds.ShootingGrounds import ShootingGrounds
from A3CBootcampGame.MultiDuelGrounds.MultiDuelGrounds import MultiDuelGrounds

class Settings():
    def __init__(self):
        # General settings:
        self.version = "1.20"
        self.agentName = "A3CMaster"
        self.activity = "MDGLSTM"
        self.gpuMemoryFraction = 1.0

        # Game settings:
        self.game = "MultiDuelGrounds"
        self.screenRes = 1000 # Screen is always a square
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 60 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame

        self.quadTreeDepth = 3
        self.quadTreeMaxObjects = 6
        self.renderQuads = True

        self.games = {"FeedingGrounds": [FeedingGrounds, 4],
                      "ShootingGrounds": [ShootingGrounds, 9],
                      "MultiDuelGrounds": [MultiDuelGrounds, 9]}

        # If enabled, the game will assign rewards to the state-action-reward tuple that caused the reward.
        # If a bullet is shot at time step 10, and it hits a enemy at time step 15, the time step when the bullet
        # was shot will receive the reward, because that's the time step when the action causing the reward happened.
        self.causalityTracking = False

        # AI settings:
        self.trainingEpisodes = 1000000

        # Hyper parameters:
        self.model = "ACNetwork"

        self.gameRes = 80
        self.actionSize = self.games[self.game][1]
        self.gamma = 0.99
        self.trainerCount = 8
        self.workersPerTrainer = 4
        self.maxEpisodeLength = 1200 # Does not effect fixed episode length games (Feeding/Shooting grounds)
        self.bootStrapCutOff = 100
        self.learningRate = 0.8e-4
        self.lrDecayRate = 0.95
        self.lrDecayStep = 100
        self.entropyWeight = 0.01
        self.valueWeight = 0.5
        self.deepRLRate = 4 # how many frames to wait for sampling experiences for deepRLAgent, and updating the agent

        self.loadCheckpoint = False
        self.saveCheckpoint = True
        self.logSummaries = True
        self.train = True
        self.tfCheckpoint = 'C:/deepRLAgent/Agent/MultiDuelGrounds_ACNetworkLSTM_LSTMTest8e-6LR_flat_2SampleRate_A3CLSTM_1.01A3CLSTM-14219'  # Check point to load
        self.tfGraphPath = 'C:/deepRLAgent/Agent/{}_{}_{}_{}_{}'.format(self.game, self.model, self.activity, self.agentName, self.version)
        self.tbPath = 'C:/deepRLAgent/tensorboard/{}/{}/{}/{}/{}'.format(self.game, self.model, self.activity, self.agentName, self.version)
        self.gifPath = 'C:/deepRLAgent/gif/{}_{}'.format(self.agentName, self.version)
