class GameHandler:
    def __init__(self, player, enemyHandler):
        self.player = player
        self.enemyHandler = enemyHandler
        self.playerScore = 0

    def update(self, events, episodeData, bootStrapCounter, bootStrapCutOff):
        gameInProgress = True
        while len(events):
            event = events.pop()
            if "enemy" in event and "bullet" in event: # Assign reward
                episodeData[event["timeStep"] - bootStrapCounter*bootStrapCutOff][2] = 1
                self.playerScore += 1
            elif "player" in event and "bullet" in event:
                episodeData[event["timeStep"] - bootStrapCounter*bootStrapCutOff][2] = -1
                gameInProgress = False
        return gameInProgress

    def resetGame(self):
        self.playerScore = 0
        self.enemyHandler.reset()
        self.player.reset()