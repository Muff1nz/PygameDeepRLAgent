class GameHandler:
    def __init__(self, player, enemyHandler):
        self.player = player
        self.enemyHandler = enemyHandler
        self.playerScore = 0

    def update(self, events, episodeData, bootStrapCounter, bootStrapCutOff):
        gameInProgress = True
        while len(events):
            event = events.pop()
            if event[0] == "Player hit enemy!": # Assign reward
                episodeData[event[1] - bootStrapCounter*bootStrapCutOff][2] = 1
                self.playerScore += 1
            elif event[0] == "Enemy hit player!":
                episodeData[event[1] - bootStrapCounter*bootStrapCutOff][2] = -1
                gameInProgress = False
        return gameInProgress

    def resetGame(self):
        self.playerScore = 0
        self.enemyHandler.reset()
        self.player.reset()