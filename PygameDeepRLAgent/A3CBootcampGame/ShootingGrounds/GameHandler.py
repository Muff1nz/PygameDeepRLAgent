import tensorflow as tf

class GameHandler:
    def __init__(self, events, player, target):
        self.events = events
        self.player = player
        self.target = target
        self.playerScore = 0
        self.episodeLength = 1200 # Amount of frames in a episode

    def update(self, timeStep, episodeData):
        reset = False # So that all events get checked if terminal state
        while len(self.events):
            event = self.events.pop()
            if event[0] == "Player hit target!": # Assign revard
                episodeData[event[1]][2] = 1
                self.playerScore += 1
        if timeStep == self.episodeLength:
            return False
        return True

    def resetGame(self):
        self.playerScore = 0
        self.target.reset()
        self.player.reset()