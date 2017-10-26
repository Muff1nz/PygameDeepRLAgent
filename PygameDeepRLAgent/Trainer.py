import numpy as np
import scipy.signal
import tensorflow as tf
import imageio
import os

from ACNetwork import ACNetwork


class Trainer():
    def __init__(self, settings, sess, number, coord, globalEpisodes):
        self.settings = settings
        self.coord = coord
        self.sess = sess
        self.name = 'trainer{}'.format(number)
        self.number = number

        self.globalEpisodes = globalEpisodes
        self.incrementGE = self.globalEpisodes.assign_add(1)
        self.localEpisodes = tf.Variable(0, dtype=tf.int32, name='local_episodes', trainable=False)
        self.incrementLE = self.localEpisodes.assign_add(1)

        self.localSteps = tf.Variable(0, dtype=tf.int32, name='{}_episodes'.format(self.name), trainable=False)
        self.writer = tf.summary.FileWriter(settings.tbPath + self.name)
        self.summaryData = {}

        self.localAC = ACNetwork(settings, self.name, step=self.localSteps)
        globalNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        localNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.updateLocalVars = []
        for gnVars, lnVars in zip(globalNetwork, localNetwork):
            self.updateLocalVars.append(lnVars.assign(gnVars))

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self, trainingData):
        episodeData = trainingData["episodeData"]
        values = trainingData["values"]
        bootStrapValue = trainingData["bootStrapValue"]
        score = trainingData["score"]
        worker = trainingData["worker"]

        print("{} is training with worker{}s data!".format(self.name, worker))

        frames = episodeData[:, 0]
        actions = episodeData[:, 1]
        rewards = episodeData[:, 2]
        frames = np.asarray(frames.tolist())
        size = len(episodeData)
        gamma = self.settings.gamma

        rewardsPlus = np.asarray(rewards.tolist() + [bootStrapValue])
        discountedRewards = self.discount(rewardsPlus, gamma)[:-1]
        valuePlus = np.asarray(values + [bootStrapValue])
        # Calculates the generalized advantage estimate
        advantages = rewards + gamma * valuePlus[1:] - valuePlus[:-1]
        advantages = self.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        self.sess.run(self.updateLocalVars)
        feedDict = {self.localAC.targetV: discountedRewards,
                    self.localAC.frame: frames,
                    self.localAC.actions: actions,
                    self.localAC.advantages: advantages}
        vl, pl, e, gn, vn, _ = self.sess.run([self.localAC.valueLoss,
                                              self.localAC.policyLoss,
                                              self.localAC.entropy,
                                              self.localAC.gradNorms,
                                              self.localAC.varNorms,
                                              self.localAC.applyGradsGlobal],
                                              feed_dict=feedDict)

        #if (e/size < 0.01):
        #    print("Model collapses, entropy: {}".format(e/size))
        #    self.coord.request_stop()

        if (not worker in self.summaryData):
            self.summaryData[worker] = SummaryData(self.writer)
        self.summaryData[worker].extend(size, vl, pl, e, gn, vn, score)
        if bootStrapValue == 0:  # This means that a worker has finished an episode
            self.sess.run(self.incrementGE)
            self.sess.run(self.incrementLE)
            if self.sess.run(self.localEpisodes) % self.settings.logFreq == 0:
                self.summaryData[worker].write(self.sess.run(self.localAC.lr), self.sess.run(self.localEpisodes))
            self.summaryData[worker].clear()

        if (self.sess.run(self.localEpisodes) % 100 == 0):
            self.writeFramesToDisk(frames)

    def writeFramesToDisk(self, frames):
        print("{} is saving frames to disk".format(self.name))

        path = "{}/{}/{}/".format(self.settings.imagePath, self.name, self.sess.run(self.localEpisodes))
        if not os.path.exists(path):
            print("{} is making path: {}".format(self.name, path))
            os.makedirs(path)
        framePick = np.random.randint(0, len(frames))
        frameSeq = frames[framePick]
        for i in range(self.settings.frameSequenceLen):
            imageio.imwrite(path + "{}.png".format(i), frameSeq[i])

class SummaryData:
    def __init__(self, writer):
        self.clear()
        self.writer = writer

    def extend(self, size, vl, pl, e, gn, vn, score):
        self.size += size
        self.valueLoss += vl
        self.policyLoss += pl
        self.entropy += e
        self.gradientNorm += gn
        self.variableNorm += vn
        self.score += score
        self.bootStrapCount += 1

    def clear(self):
        self.size = 0
        self.valueLoss = 0
        self.policyLoss = 0
        self.entropy = 0
        self.gradientNorm = 0
        self.variableNorm = 0
        self.score = 0
        self.bootStrapCount = 0

    def write(self, lr, episode):
        summary = tf.Summary()
        summary.value.add(tag="Performance/Score", simple_value=self.score)
        summary.value.add(tag="Performance/BootStrapCount", simple_value=self.bootStrapCount)
        summary.value.add(tag='Losses/Value Loss', simple_value=float(self.valueLoss/self.size))
        summary.value.add(tag='Losses/Policy Loss', simple_value=float(self.policyLoss/self.size))
        summary.value.add(tag='Losses/Entropy', simple_value=float(self.entropy/self.size))
        summary.value.add(tag='Losses/Grad Norm', simple_value=float(self.gradientNorm / self.bootStrapCount))
        summary.value.add(tag='Losses/Var Norm', simple_value=float(self.variableNorm / self.bootStrapCount))
        summary.value.add(tag='Losses/Learning rate',
                          simple_value=float(lr))
        self.writer.add_summary(summary, episode)
        self.writer.flush()