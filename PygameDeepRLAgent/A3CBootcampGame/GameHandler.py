import tensorflow as tf

class GameHandler:
    def __init__(self, events, player, food, episodeData):
        self.events = events
        self.player = player
        self.food = food
        self.playerScore = 0
        self.episodeLength = 900 # Amount of frames in a episode

    def update(self, timeStep, episodeData, bootStrapCounter, bootStrapCutOff):
        reset = False # So that all events get checked if terminal state
        while len(self.events):
            event = self.events.pop()
            if event[0] == "Player ate food!":
                episodeData[event[1] - bootStrapCounter*bootStrapCutOff][2] = 1
                self.playerScore += 1
        if timeStep == self.episodeLength:
            return False
        return True

    def resetGame(self):
        self.playerScore = 0
        self.food.reset()
        self.player.reset()