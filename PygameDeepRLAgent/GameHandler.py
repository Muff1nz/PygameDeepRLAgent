import tensorflow as tf

class GameHandler:
    def __init__(self, events, em, player, episodeData):
        self.events = events
        self.player = player
        self.em = em
        self.playerScore = 0
        self.episodeCount = 0

    def update(self, episodeData, bootStrapCounter, bootStrapCutOff):
        reset = False # So that all events get checked if terminal state
        while len(self.events):
            event = self.events.pop()
            if event[0] == "Player killed":
                episodeData[event[1] - bootStrapCounter*bootStrapCutOff][2] = -1
                reset = True
            if event[0] == "Enemy killed":
                episodeData[event[1] - bootStrapCounter*bootStrapCutOff][2] = 1
                self.playerScore += 1
        if reset:
            return False
        return True

    def resetGame(self):
        self.episodeCount += 1
        self.playerScore = 0
        self.em.reset()
        self.player.reset()