
class GameHandler:
    def __init__(self, events, em, player):
        self.events = events
        self.em = em
        self.player = player
        self.playerScore = 0

    def update(self, replayMemory):
        reset = False # So that all events get checked if terminal state
        while len(self.events):
            event = self.events.pop()
            #print(event)
            if event[0] == "Player killed":
                replayMemory.experienceMemory[event[1]] = -1 # Assign reward to experience
                reset = True
            if event[0] == "Enemy killed":
                replayMemory.experienceMemory[event[1]] = 1 # Assign reward to experience
                self.playerScore += 1
        if reset:
            self.resetGame()

    def resetGame(self):
        self.playerScore = 0
        self.em.reset()
        self.player.reset()