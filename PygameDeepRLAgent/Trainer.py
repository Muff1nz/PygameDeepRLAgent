from queue import Queue
from threading import Thread
from multiprocessing import Queue as PQueue, Process

import numpy as np
import scipy.signal
import tensorflow as tf

from ACNetworkLSTM import ACNetworkLSTM
from Worker import Worker
import GameRunner

class Trainer(Thread):
<<<<<<< HEAD
    def __init__(self, settings, models, number, coord, globalEpisodes):
=======
    def __init__(self, settings, sess, number, coord, globalEpisodes):
>>>>>>> origin/master
        Thread.__init__(self)
        self.settings = settings
        self.trainerQueue = Queue(5)
        self.coord = coord
        self.name = 'trainer{}'.format(number)
        self.number = number

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

<<<<<<< HEAD
    def init(self, sess):
        self.sess = sess
        workers = []
        for i in range(self.settings.workersPerTrainer):
            workers.append(Worker(self.settings, self.sess, self.name, i, self.localAC, self.trainerQueue, self.coord))
            workers[i].start()

    def isAlive(self, workers):
        for worker in workers:
            if worker.is_alive():
                return True
        return False
=======
    def run(self):
        #===INIT=====
        workers = []
        gameDataQueues = [] # used by workers and games
        playerActionQueues = [] # used by workers and games
        for i in range(self.settings.workersPerTrainer): # Set up all the workers
            gameDataQueue = PQueue()
            playerActionQueue = PQueue()
            queues = {"trainer": self.trainerQueue,
                      "gameData": gameDataQueue,
                      "playerAction": playerActionQueue}
            workers.append(Worker(self.settings, self.sess, self.name, i, self.localAC, queues, self.coord))
            gameDataQueues.append(gameDataQueue)
            playerActionQueues.append(playerActionQueue)
            self.summaryData[str(i)] = SummaryData(self.writer)
        gameProcess = Process(target=(GameRunner.run), args=(self.settings, gameDataQueues, playerActionQueues))
        #===RUN====
        gameProcess.start()
        while not self.coord.should_stop():
            self.train()
            for worker in workers:
                worker.work()
        #===CLEANUP====
        print("{} is quitting!".format(self.name))
        gameProcess.terminate()
>>>>>>> origin/master

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self):
<<<<<<< HEAD
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
        ##self.writeSummaries(vl/size, pl/size, e/size, gn, vn, score)



    def writeSummaries(self, vl, pl, e, gn, vn, score):
=======
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
            if (e/size < 0.01):
                print("Model collapses, entropy: {}".format(e/size))
                self.coord.request_stop()

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
>>>>>>> origin/master
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