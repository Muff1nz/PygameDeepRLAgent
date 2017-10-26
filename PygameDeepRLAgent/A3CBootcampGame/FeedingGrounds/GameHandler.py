import tensorflow as tf

class GameHandler:
    def __init__(self, player, food):
        self.player = player
        self.food = food
        self.playerScore = 0
        self.episodeLength = 1200 # Amount of frames in a episode
        self.timer = 0

    def update(self, events, gameCounter, episodeData, bootStrapCounter, bootStrapCutOff):
        self.timer += 1
        while len(events):
            event = events.pop()
            if "player" in event and "food" in event:
                episodeData[event["timeStep"] - bootStrapCounter * bootStrapCutOff][2] = 0.5
                self.playerScore += 1
        if gameCounter == self.episodeLength:
            return False
        return True

    def resetGame(self):
        self.playerScore = 0
        self.food.reset()
        self.player.reset()