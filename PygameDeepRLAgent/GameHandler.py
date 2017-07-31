import tensorflow as tf

class GameHandler:
    def __init__(self, events, em, player, writer, sess):
        self.events = events
        self.em = em
        self.player = player
        self.playerScore = 0
        self.episodeCount = 0
        self.writer = writer
        self.sess = sess
        self.scorePlaceholder = tf.placeholder('int64')
        self.epsiodeScoreSummary = tf.summary.scalar('episodeScore', self.scorePlaceholder)

    def update(self, replayMemory):
        reset = False # So that all events get checked if terminal state
        while len(self.events):
            event = self.events.pop()
            if event[0] == "Player killed":
                replayMemory.experienceMemory[event[1]] = -2 # Assign reward to experience
                reset = True
            if event[0] == "Enemy killed":
                replayMemory.experienceMemory[event[1]] = 1 # Assign reward to experience
                self.playerScore += 1
        if reset:
            self.resetGame()

    def resetGame(self):
        self.episodeCount += 1
        summary = self.sess.run(self.epsiodeScoreSummary, feed_dict={self.scorePlaceholder: self.playerScore})
        self.writer.add_summary(summary, self.episodeCount)
        self.playerScore = 0
        self.em.reset()
        self.player.reset()