from queue import Queue
from threading import Thread

import numpy as np
import scipy.signal
import tensorflow as tf

from ACNetworkLSTM import ACNetworkLSTM
from Worker import Worker

import time


class Trainer(Thread):
    def __init__(self, settings, sess, number, coord, globalEpisodes):
        Thread.__init__(self)
        self.settings = settings
        self.trainerQueue = Queue(5)
        self.coord = coord
        self.sess = sess
        self.name = 'trainer{}'.format(number)

        with tf.device("/cpu:0"):
            self.globalEpisodes = globalEpisodes
            self.incrementGE = self.globalEpisodes.assign_add(1)
            self.localEpisodes = tf.Variable(0, dtype=tf.int32, name='local_episodes', trainable=False)
            self.incrementLE = self.localEpisodes.assign_add(1)

        self.localSteps = tf.Variable(0, dtype=tf.int32, name='{}_episodes'.format(self.name), trainable=False)
        self.writer = tf.summary.FileWriter(settings.tbPath + self.name)
        self.summaryData = {}

        self.localAC = ACNetworkLSTM(settings, self.name, step=self.localSteps)
        globalNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        localNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.updateLocalVars = []
        for gnVars, lnVars in zip(globalNetwork, localNetwork):
            self.updateLocalVars.append(lnVars.assign(gnVars))

    def run(self):
        workers = []
        for i in range(self.settings.workersPerTrainer):
            workers.append(Worker(self.settings, self.sess, self.name, i, self.localAC, self.trainerQueue, self.coord))
            self.summaryData[str(i)] = SummaryData(self.writer)
        while not self.coord.should_stop():
            self.train()
            for worker in workers:
                worker.work()
        for worker in workers:
            worker.stop()
        print("{} is quitting!".format(self.name))

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self):
        if not self.trainerQueue.empty():
            trainingData = self.trainerQueue.get()
            episodeData = trainingData["episodeData"]
            values = trainingData["values"]
            bootStrapValue = trainingData["bootStrapValue"]
            score = trainingData["score"]
            worker = trainingData["worker"]

            frames = episodeData[:, 0]
            actions = episodeData[:, 1]
            rewards = episodeData[:, 2]
            frames = np.asarray(frames.tolist())
            size = len(episodeData)
            gamma = self.settings.gamma

            self.rewardsPlus = np.asarray(rewards.tolist() + [bootStrapValue])
            discountedRewards = self.discount(self.rewardsPlus, gamma)[:-1]
            self.valuePlus = np.asarray(values + [bootStrapValue])
            # Calculates the generalized advantage estimate
            advantages = rewards + gamma * self.valuePlus[1:] - self.valuePlus[:-1]
            advantages = self.discount(advantages, gamma)

            # Update the global network using gradients from loss
            # Generate network statistics to periodically save
            self.sess.run(self.updateLocalVars)
            rnnState = self.localAC.stateInit
            feedDict = {self.localAC.targetV: discountedRewards,
                        self.localAC.frame: frames,
                        self.localAC.actions: actions,
                        self.localAC.advantages: advantages,
                        self.localAC.stateIn[0]: rnnState[0],
                        self.localAC.stateIn[1]: rnnState[1]}
            vl, pl, e, gn, vn, _ = self.sess.run([self.localAC.valueLoss,
                                                  self.localAC.policyLoss,
                                                  self.localAC.entropy,
                                                  self.localAC.gradNorms,
                                                  self.localAC.varNorms,
                                                  self.localAC.applyGradsGlobal],
                                                  feed_dict=feedDict)

            self.summaryData[str(worker)].extend(size, vl, pl, e, gn, vn, score)
            if bootStrapValue == 0:  # This means that a worker has finished an episode
                self.sess.run(self.incrementGE)
                self.sess.run(self.incrementLE)
                print("{} is training with worker{}s data!".format(self.name, worker))
                if self.sess.run(self.localEpisodes) % self.settings.logFreq == 0:
                    self.summaryData[str(worker)].write(self.sess.run(self.localAC.lr), self.sess.run(self.localEpisodes))
                self.summaryData[str(worker)].clear()

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



    '''
    def makeGifs(self, episodeData, episodeCount, settings):
        plusRewardIndex = -1
        minusRewardIndex = -1
        giflen = 10
        for i, episode in enumerate(episodeData):
            if i > giflen and i < len(episodeData) - giflen:
                if episode[2] > 0:
                    plusRewardIndex = i
                if episode[2] < 0:
                    minusRewardIndex = i
                if minusRewardIndex != -1 and plusRewardIndex != -1:
                    break
        frames = episodeData[:, 0]
        frames = np.asarray(frames.tolist())
        if plusRewardIndex != -1:
            imageio.mimsave(settings.gifPath + "_plus_" + settings.activity + "_" + str(episodeCount) + ".gif",
                            frames[(plusRewardIndex - giflen):(plusRewardIndex + giflen)])
        if minusRewardIndex != -1:
            imageio.mimsave(settings.gifPath + "_minus_" + settings.activity + "_" + str(episodeCount) + ".gif",
                            frames[(minusRewardIndex - giflen):(minusRewardIndex + giflen)])
    '''