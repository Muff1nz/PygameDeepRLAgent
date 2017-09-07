import tensorflow as tf
import numpy as np
import scipy.signal
import imageio

from ACNetwork import ACNetwork

class Worker:
    def __init__(self, settings, i):
        self.name = "worker" + str(i)
        self.number = i
        self.settings = settings
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
        # Calculates the generalized advantage estimate
        advantages = rewards + gamma * self.valuePlus[1:] - self.valuePlus[:-1]
        advantages = self.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feedDict = { self.localAC.targetV: discountedRewards,
                     self.localAC.frame: frames,
                     self.localAC.actions: actions,
                     self.localAC.advantages: advantages,
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
            sess.run(self.updateLocalVars)
            while not coord.should_stop():
                values = []
                episodeInProgress = True
                episodeCount = sess.run(self.localEpisodes)
                while episodeInProgress:
                    gameData = gameDataQueue.get() # Get data from the game

                    if gameData[0] == "CurrentFrame": # Process the next action based on frame
                        frame = gameData[1]
                        feedDict = {self.localAC.frame: [frame]}
                        actionDist, value = sess.run([self.localAC.logits, self.localAC.value], feed_dict=feedDict)
                        action = np.random.choice(actionDist[0], p=actionDist[0])
                        action = np.argmax(actionDist==action)

                        playerActionQueue.put(action)
                        values.append(value[0,0])

                    elif gameData[0] == "Bootstrap": # Bootstrap from bootstrap data
                        print("{} is bootstrapping!".format(self.name))
                        bootstrapValues = values[0:settings.bootStrapCutOff]
                        values = values[settings.bootStrapCutOff::]
                        episodeData = np.array(gameData[1])
                        self.train(episodeData, bootstrapValues, sess, settings.gamma, values[0])

                    elif gameData[0] == "EpisodeData": # Episode is finished, perform training and logging
                        if settings.train:
                            print("{} is training!".format(self.name))
                            episodeData = np.array(gameData[1])
                            sess.run(self.updateLocalVars)
                            vl, pl, e, gn, vn = self.train(episodeData, values, sess, settings.gamma)
                            score = gameDataQueue.get()[1]
                            episodeInProgress = False
                            values = []
                            # Log summaries
                            if settings.logSummaries and not episodeCount % 5:
                                summary = tf.Summary()
                                summary.value.add(tag="Performance/Score", simple_value=score)
                                summary.value.add(tag='Losses/Value Loss', simple_value=float(vl))
                                summary.value.add(tag='Losses/Policy Loss', simple_value=float(pl))
                                summary.value.add(tag='Losses/Entropy', simple_value=float(e))
                                summary.value.add(tag='Losses/Grad Norm', simple_value=float(gn))
                                summary.value.add(tag='Losses/Var Norm', simple_value=float(vn))
                                summary.value.add(tag='Losses/Learning rate',
                                                  simple_value=float(sess.run(self.localAC.lr)))
                                self.writer.add_summary(summary, episodeCount)
                                self.writer.flush()
                            # Save model and make a gif, if you're worker 0
                            if self.number == 0:
                                print(sess.run(self.localEpisodes))
                                if not episodeCount % 500 and episodeCount > 0:
                                    print("worker0 is saving the tf graph!")
                                    if settings.saveCheckpoint:
                                        saver.save(sess, settings.tfGraphPath + settings.agentName, self.localEpisodes)
                                    if settings.logSummaries:
                                        print("Worker0 saving a gif!")
                                        makeGifs(episodeData, episodeCount, settings)
                                if episodeCount > settings.trainingEpisodes:
                                    print("Training finished, saving and quitting program!".format(self.name))
                                    saver.save(sess, settings.tfGraphPath + settings.agentName, self.localEpisodes)
                                    coord.request_stop()
                        else:
                            gameDataQueue.get()

                    elif gameData[0] == "Game closed!": # Game has been closed
                        print("{}s game closed, saving and quitting program!".format(self.name))
                        saver.save(sess, settings.tfGraphPath + settings.agentName, self.localEpisodes)
                        coord.request_stop()

                    else:
                        print("Invalid game data! got: {}".format(gameData[0]))
                episodeCount += 1
            print("{} is quitting!".format(self.name))



def makeGifs(episodeData, episodeCount, settings):
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