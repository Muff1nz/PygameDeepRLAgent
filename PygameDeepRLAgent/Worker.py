import tensorflow as tf
import numpy as np
import scipy.signal

from ACNetwork import ACNetwork

class Worker:
    def __init__(self, settings, i, trainer):
        self.name = "worker" + str(i)
        self.settings = settings
        self.trainer = trainer
        self.writer = tf.summary.FileWriter(settings.tbPath + self.name)

        self.localAC = ACNetwork(settings, self.name, trainer)
        globalNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        localNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.updateLocalVars = []
        for gnVars, lnVars in zip(globalNetwork, localNetwork):
            self.updateLocalVars.append(lnVars.assign(gnVars))

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self, episodeData, values, sess, gamma, bootStrapValue = 0.0):
        episodeData = np.array(episodeData)
        values = np.array(values)
        frames = episodeData[:, 0]
        actions = episodeData[:, 1]
        rewards = episodeData[:, 2]
        frames = np.asarray(frames.tolist())


        self.rewardsPlus = np.asarray(rewards.tolist() + [bootStrapValue])
        discountedRewards = self.discount(self.rewardsPlus, gamma)[:-1]
        self.valuePlus = np.asarray(values.tolist() + [bootStrapValue])
        advantages = rewards + gamma * self.valuePlus[1:] - self.valuePlus[:-1]
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
        episodeCount = 0
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
                    elif gameData[0] == "Bootstrap":
                        print("{} is bootstrapping!".format(self.name))
                        frame = gameData[2]
                        bootStrapValue = sess.run(self.localAC.value, # Compute value of most recent state
                            feed_dict = {self.localAC.processedFrame: [frame],
                                         self.localAC.stateIn[0]: rnnState[0],
                                         self.localAC.stateIn[1]: rnnState[1]})[0, 0]
                        bootstrapValues = episodeValues[0:settings.bootStrapCutOff]
                        episodeValues = episodeValues[settings.bootStrapCutOff::]
                        self.train(gameData[1], bootstrapValues, sess, settings.gamma, bootStrapValue)
                    elif gameData[0] == "EpisodeData":
                        print("{} is training!".format(self.name))
                        self.train(gameData[1], episodeValues, sess, settings.gamma)
                        gameData = gameDataQueue.get()
                        summary = tf.Summary()
                        summary.value.add(tag="Score", simple_value=gameData[1])
                        self.writer.add_summary(summary, episodeCount)
                        episodeInProgress = False
                        episodeValues = []
                    else:
                        print("Invalid game data!")
                episodeCount += 1

