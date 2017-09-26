import tensorflow as tf

class GameHandler:
    def __init__(self, events, player, food):
        self.events = events
        self.player = player
        self.food = food
        self.playerScore = 0
        self.episodeLength = 1200 # Amount of frames in a episode

    def update(self, timeStep, episodeData):
        while len(self.events):
            event = self.events.pop()
            if event[0] == "Player ate food!":
                episodeData[event[1]][2] = 0.5
                self.playerScore += 1
        if timeStep == self.episodeLength:
            return False
        return True

    def resetGame(self):
        self.playerScore = 0
        self.food.reset()
        self.player.reset()