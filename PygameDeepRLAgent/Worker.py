import tensorflow as tf
import numpy as np
import scipy.signal
import imageio

from ACNetwork import ACNetwork

class Worker:
    def __init__(self, settings, i, trainer, globalEpisodes):
        self.name = "worker" + str(i)
        self.number = i
        self.settings = settings
        self.globalEpisodes = globalEpisodes
        self.increment = self.globalEpisodes.assign_add(1)
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
        values = np.array(values)
        frames = episodeData[:, 0]
        actions = episodeData[:, 1]
        rewards = episodeData[:, 2]
        frames = np.asarray(frames.tolist())
        size = len(episodeData)

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
        vl, pl, e, gn, vn, _ = sess.run([ self.localAC.valueLoss,
                                          self.localAC.policyLoss,
                                          self.localAC.entropy,
                                          self.localAC.gradNorms,
                                          self.localAC.varNorms,
                                          self.localAC.applyGrads],
                                          feed_dict=feedDict)
        return vl / size, pl / size, e / size, gn, vn

    def work(self, settings, gameDataQueue, playerActionQueue, sess, coord, saver):
        episodeCount = sess.run(self.globalEpisodes)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.updateLocalVars)
                episodeValues = []
                rnnState = self.localAC.stateInit
                episodeInProgress = True
                episodeStepCount = 0
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
                        episodeStepCount += 1
                    elif gameData[0] == "Bootstrap":
                        print("{} is bootstrapping!".format(self.name))
                        bootstrapValues = episodeValues[0:settings.bootStrapCutOff]
                        episodeValues = episodeValues[settings.bootStrapCutOff::]
                        episodeData = np.array(gameData[1])
                        self.train(episodeData, bootstrapValues, sess, settings.gamma, episodeValues[0])
                        sess.run(self.updateLocalVars)
                    elif gameData[0] == "EpisodeData":
                        print("{} is training!".format(self.name))
                        episodeData = np.array(gameData[1])
                        vl, pl, e, gn, vn = self.train(episodeData, episodeValues, sess, settings.gamma)
                        gameData = gameDataQueue.get()

                        summary = tf.Summary()
                        summary.value.add(tag="Performance/Score", simple_value=gameData[1])
                        summary.value.add(tag="Performance/Length", simple_value=float(episodeStepCount))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(vl))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(pl))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(gn))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(vn))
                        self.writer.add_summary(summary, episodeCount)
                        self.writer.flush()

                        episodeInProgress = False
                        episodeValues = []
                        if self.number == 0:
                            sess.run(self.increment)
                            print(sess.run(self.globalEpisodes))
                            if not episodeCount % 100:
                                print("worker0 is saving the tf graph!")
                                saver.save(sess, settings.tfGraphPath + settings.agentName, self.globalEpisodes)
                                print("Worker0 saving a gif!")
                                frames = episodeData[:, 0]
                                frames = np.asarray(frames.tolist())
                                if len(episodeData) > 10:
                                    frames = frames[-10:]
                                imageio.mimsave(settings.gifPath + str(episodeCount) + ".gif", frames)
                    elif gameData[0] == "Game closed!":
                        print("{}s game closed, saving and quitting program!".format(self.name))
                        saver.save(sess, settings.tfGraphPath + settings.agentName, self.globalEpisodes)
                        coord.request_stop()
                    else:
                        print("Invalid game data!")
                episodeCount += 1

