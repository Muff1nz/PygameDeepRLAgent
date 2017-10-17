class GameHandler:
    def __init__(self, player, target):
        self.player = player
        self.target = target
        self.playerScore = 0
        self.episodeLength = 1200 # Amount of frames in a episode

    def update(self, events, timeStep, episodeData):
        while len(events):
            event = events.pop()
            if "target" in event and "bullet" in event:
                episodeData[event["timeStep"]][2] = 1 # enemy was hit by bullet
                self.playerScore += 1
        if timeStep == self.episodeLength:
            return False
        return True

    def resetGame(self):
        self.playerScore = 0
        self.target.reset()
        self.player.reset()