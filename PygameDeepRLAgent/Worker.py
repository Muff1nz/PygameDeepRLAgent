import tensorflow as tf
import numpy as np
import scipy.signal
import imageio

from ACNetwork import ACNetwork

class Worker:
    def __init__(self, settings, i):
        self.name = "worker" + str(i)
        self.number = i
        self.localEpisodes = tf.Variable(0, dtype=tf.int32, name='{}_episodes'.format(self.name), trainable=False)
        self.writer = tf.summary.FileWriter(settings.tbPath + self.name)

        self.localAC = ACNetwork(settings, self.name, step=self.localEpisodes)
        globalNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        localNetwork = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.updateLocalVars = []
        for gnVars, lnVars in zip(globalNetwork, localNetwork):
            self.updateLocalVars.append(lnVars.assign(gnVars))

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self, episodeData, values, sess, gamma, bootStrapValue=0):
        frames = episodeData[:, 0]
        actions = episodeData[:, 1]
        rewards = episodeData[:, 2]
        frames = np.asarray(frames.tolist())
        size = len(episodeData)

        self.rewardsPlus = np.asarray(rewards.tolist() + [bootStrapValue])
        discountedRewards = self.discount(self.rewardsPlus, gamma)[:-1]
        self.valuePlus = np.asarray(values + [bootStrapValue])
        advantages = rewards + gamma * self.valuePlus[1:] - self.valuePlus[:-1]
        advantages = self.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        #rnnIn = self.localAC.stateInit
        feedDict = { self.localAC.targetV: discountedRewards,
                     self.localAC.processedFrame: frames,
                     self.localAC.actions: actions,
                     self.localAC.advantages: advantages,
                     #self.localAC.stateIn[0]: rnnIn[0],
                     #self.localAC.stateIn[1]: rnnIn[1]
                     }
        vl, pl, e, gn, vn, _ = sess.run([ self.localAC.valueLoss,
                                          self.localAC.policyLoss,
                                          self.localAC.entropy,
                                          self.localAC.gradNorms,
                                          self.localAC.varNorms,
                                          self.localAC.applyGradsGlobal],
                                          feed_dict=feedDict)
        return vl / size, pl / size, e / size, gn, vn

    def work(self, settings, gameDataQueue, playerActionQueue, sess, coord, saver):
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                values = []
                episodeInProgress = True
                episodeCount = sess.run(self.localEpisodes)
                sess.run(self.updateLocalVars)
                #rnnState = self.localAC.stateInit
                while episodeInProgress:
                    gameData = gameDataQueue.get()
                    if gameData[0] == "CurrentFrame":
                        frame = gameData[1]
                        feedDict = {self.localAC.processedFrame: [frame],
                                    #self.localAC.stateIn[0]: rnnState[0],
                                    #self.localAC.stateIn[1]: rnnState[1]
                                    }
                        actionDist, value = sess.run([self.localAC.logits,
                                                      self.localAC.value,
                                                      #self.localAC.stateOut
                                                      ],
                                                      feed_dict=feedDict)
                        action = np.random.choice(actionDist[0], p=actionDist[0])
                        action = np.argmax(actionDist==action)

                        playerActionQueue.put(action)
                        values.append(value[0,0])
                    elif gameData[0] == "Bootstrap":
                        print("{} is bootstrapping!".format(self.name))
                        bootstrapValues = values[0:settings.bootStrapCutOff]
                        values = values[settings.bootStrapCutOff::]
                        episodeData = np.array(gameData[1])
                        self.train(episodeData, bootstrapValues, sess, settings.gamma, values[0])
                    elif gameData[0] == "EpisodeData":
                        print("{} is training!".format(self.name))
                        episodeData = np.array(gameData[1])
                        vl, pl, e, gn, vn = self.train(episodeData, values, sess, settings.gamma)
                        score = gameDataQueue.get()[1]

                        summary = tf.Summary()
                        summary.value.add(tag="Performance/Score", simple_value=score)
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(vl))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(pl))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(gn))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(vn))
                        self.writer.add_summary(summary, episodeCount)
                        self.writer.flush()

                        episodeInProgress = False
                        values = []
                        if self.number == 0:
                            print(sess.run(self.localEpisodes))
                            if not episodeCount % 100:
                                print("worker0 is saving the tf graph!")
                                saver.save(sess, settings.tfGraphPath + settings.agentName, self.localEpisodes)
                            if not episodeCount % 50:
                                print("Worker0 saving a gif!")
                                makeGifs(episodeData, episodeCount, settings)
                    elif gameData[0] == "Game closed!":
                        print("{}s game closed, saving and quitting program!".format(self.name))
                        saver.save(sess, settings.tfGraphPath + settings.agentName, self.localEpisodes)
                        coord.request_stop()
                    else:
                        print("Invalid game data!")
                episodeCount += 1

def makeGifs(episodeData, episodeCount, settings):
    plusRewardIndex = -1
    minusRewardIndex = -1
    for i, episode in enumerate(episodeData):
        if i > 11 and i < len(episodeData) - 11:
            if episode[2] > 0:
                plusRewardIndex = i
            if episode[2] < 0:
                minusRewardIndex = i
            if minusRewardIndex != -1 and plusRewardIndex != -1:
                break
    frames = episodeData[:, 0]
    frames = np.asarray(frames.tolist())
    if plusRewardIndex != -1:
        imageio.mimsave(settings.gifPath + "_plus_" + str(episodeCount) + ".gif",
                        frames[plusRewardIndex-10:plusRewardIndex+10])
    if minusRewardIndex != -1:
        imageio.mimsave(settings.gifPath + "_minus_" + str(episodeCount) + ".gif",
                        frames[minusRewardIndex-10:minusRewardIndex+10])

