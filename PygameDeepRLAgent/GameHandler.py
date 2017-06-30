
class GameHandler:
    def __init__(self, events, em, player):
        self.events = events
        self.em = em
        self.player = player
        self.playerScore = 0

    def update(self):
        reset = False # So that all events get checked if terminal state
        while len(self.events):
            event = self.events.pop()
            if event == "Player killed":
                reset = True
            if event == "Enemy killed":
                self.playerScore += 1
        if reset:
            self.resetGame()

    def resetGame(self):
        self.playerScore = 0
        self.em.reset()
        self.player.reset()