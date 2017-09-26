import tensorflow as tf
from threading import Thread
from queue import Queue
import numpy as np
import scipy.signal

from Worker import Worker

class Trainer(Thread):
    def __init__(self, settings, sess, models, number, coord, globalEpisodes):
        Thread.__init__(self)
        self.settings = settings
        self.trainerQueue = Queue(5)
        self.coord = coord
        self.sess = sess
        self.name = 'trainer{}'.format(number)

        self.globalEpisodes = globalEpisodes
        self.incrementGE = self.globalEpisodes.assign_add(1)

        self.localEpisodes = tf.Variable(0, dtype=tf.int32, name='{}_episodes'.format(self.name), trainable=False)
        self.writer = tf.summary.FileWriter(settings.tbPath + self.name)

        self.localAC = models[settings.model](settings, self.name, step=self.localEpisodes)
        globalNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        localNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.updateLocalVars = []
        for gnVars, lnVars in zip(globalNetwork, localNetwork):
            self.updateLocalVars.append(lnVars.assign(gnVars))

    def run(self):
        workers = []
        for i in range(self.settings.workersPerTrainer):
            workers.append(Worker(self.settings, self.sess, self.name, i, self.localAC, self.trainerQueue, self.coord))
            workers[i].start()
        while not self.coord.should_stop() or self.isAlive(workers):
            self.train()
        print("{} is quitting!".format(self.name))

    def isAlive(self, workers):
        for worker in workers:
            if worker.is_alive():
                return True
        return False

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self):
        trainingData = self.trainerQueue.get()
        qSize = self.trainerQueue.qsize()
        print("{} is training! {} items left in queue".format(self.name, qSize))
        self.sess.run(self.incrementGE)
        episodeData = trainingData["episodeData"]
        values = trainingData["values"]
        bootStrapValue = trainingData["bootStrapValue"]
        score = trainingData["score"]

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
        feedDict = { self.localAC.targetV: discountedRewards,
                     self.localAC.frame: frames,
                     self.localAC.actions: actions,
                     self.localAC.advantages: advantages}
        self.sess.run(self.updateLocalVars)
        vl, pl, e, gn, vn, _ = self.sess.run([ self.localAC.valueLoss,
                                          self.localAC.policyLoss,
                                          self.localAC.entropy,
                                          self.localAC.gradNorms,
                                          self.localAC.varNorms,
                                          self.localAC.applyGradsGlobal],
                                          feed_dict=feedDict)
        self.writeSummaries(vl/size, pl/size, e/size, gn, vn, score)



    def writeSummaries(self, vl, pl, e, gn, vn, score):
        summary = tf.Summary()
        summary.value.add(tag="Performance/Score", simple_value=score)
        summary.value.add(tag='Losses/Value Loss', simple_value=float(vl))
        summary.value.add(tag='Losses/Policy Loss', simple_value=float(pl))
        summary.value.add(tag='Losses/Entropy', simple_value=float(e))
        summary.value.add(tag='Losses/Grad Norm', simple_value=float(gn))
        summary.value.add(tag='Losses/Var Norm', simple_value=float(vn))
        summary.value.add(tag='Losses/Learning rate',
                          simple_value=float(self.sess.run(self.localAC.lr)))
        self.writer.add_summary(summary, self.sess.run(self.localEpisodes))
        self.writer.flush()


    def saveModel(self):


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