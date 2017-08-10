import tensorflow as tf
import numpy as np
import scipy.signal

from ACNetwork import ACNetwork

class Worker:
    def __init__(self, settings, i, trainer):
        self.name = "worker" + str(i)
        self.settings = settings
        self.trainer = trainer

        self.localAC = ACNetwork(settings, self.name, trainer)
        globalNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        localNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.updateLocalVars = []
        for gnVars, lnVars in zip(globalNetwork, localNetwork):
            self.updateLocalVars.append(lnVars.assign(gnVars))

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self, episodeData, values, sess, gamma):
        episodeData = np.array(episodeData)
        values = np.array(values)
        frames = episodeData[:, 0]
        actions = episodeData[:, 1]
        rewards = episodeData[:, 2]
        nextFrames = episodeData[:, 3]
        frames = np.asarray(frames.tolist())


        self.rewardsPlus = np.asarray(rewards.tolist())
        discountedRewards = self.discount(self.rewardsPlus, gamma)
        self.valuePlus = np.asarray(values.tolist())
        advantages = rewards + gamma * self.valuePlus - self.valuePlus
        advantages = self.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnnState = self.localAC.stateInit
        feedDict = { self.localAC.targetV: discountedRewards,
                     self.localAC.processedFrame: frames,
                     self.localAC.actions: actions,
                     self.localAC.advantages: advantages,
                     self.localAC.stateIn[0]: rnnState[0],
                     self.localAC.stateIn[1]: rnnState[1]}
        sess.run([self.localAC.applyGrads], feed_dict=feedDict)

    def work(self, settings, gameDataQueue, playerActionQueue, sess, coord):
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.updateLocalVars)
                episodeValues = []
                rnnState = self.localAC.stateInit
                episodeInProgress = True
                while episodeInProgress:
                    gameData = gameDataQueue.get()
                    if gameData[0] == "CurrentFrame":
                        frame = gameData[1]
                        feedDict = {self.localAC.processedFrame: [frame],
                                    self.localAC.stateIn[0]: rnnState[0],
                                    self.localAC.stateIn[1]: rnnState[1]}
                        actionDist, value, rnnState = sess.run([self.localAC.policy,
                                                                self.localAC.value,
                                                                self.localAC.stateOut],
                                                                feed_dict=feedDict)
                        action = np.random.choice(actionDist[0], p=actionDist[0])
                        action = np.argmax(actionDist==action)
                        playerActionQueue.put(action)
                        episodeValues.append(value[0,0])
                    elif gameData[0] == "EpisodeData":
                        print("{} is training!".format(self.name))
                        self.train(gameData[1], episodeValues, sess, 0.99)
                        episodeInProgress = False
                    else:
                        print("Invalid game data!")

