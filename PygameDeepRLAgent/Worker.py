from multiprocessing import Queue
import numpy as np
from threading import Thread

from A3CBootcampGame.ShootingGrounds.ShootingGrounds import ShootingGrounds

class Worker(Thread):
    def __init__(self, settings, sess, trainerName, number, network, queue, coord):
        Thread.__init__(self)
        self.localAC = network
        self.name = '{}_worker{}'.format(trainerName, number)
        self.settings = settings
        self.trainerQueue = queue
        self.coord = coord
        self.sess = sess

    def run(self):
        gameDataQueue, playerActionQueue = Queue(), Queue()
        playerActionQueue.put(["WindowSettings", True if self.name == "trainer0_worker0" else False])
        game = ShootingGrounds(self.settings, gameDataQueue, playerActionQueue)
        game.start()
        while not self.coord.should_stop():
                self.work(gameDataQueue, playerActionQueue)
        print("{} is quitting!".format(self.name))
        game.terminate()

    def work(self, gameDataQueue, playerActionQueue):
        values = []
        episodeInProgress = True
        while episodeInProgress:
            gameData = gameDataQueue.get() # Get data from the game

            if gameData[0] == "CurrentFrame": # Process the next action based on frame
                frame = gameData[1]
                feedDict = {self.localAC.frame: [frame]}
                actionDist, value = self.sess.run([self.localAC.logits,
                                                       self.localAC.value],
                                                       feed_dict=feedDict)
                action = np.random.choice(actionDist[0], p=actionDist[0])
                action = np.argmax(actionDist==action)

                playerActionQueue.put(action)
                values.append(value[0,0])

            elif gameData[0] == "Bootstrap": # Bootstrap from bootstrap data
                print("{} is bootstrapping!".format(self.name))
                bootstrapValues = values[0:self.settings.bootStrapCutOff]
                values = values[self.settings.bootStrapCutOff::]
                episodeData = np.array(gameData[1])
                workerData = {"episodeData": episodeData,
                              "values": bootstrapValues,
                              "bootStrapValue": values[0],
                              "score": -1}
                self.trainerQueue.put(workerData)

            elif gameData[0] == "EpisodeData": # Episode is finished, perform training and logging
                episodeData = np.array(gameData[1])
                score = gameDataQueue.get()[1]
                workerData = {"episodeData": episodeData,
                              "values": values,
                              "bootStrapValue": 0,
                              "score": score}
                self.trainerQueue.put(workerData)
                episodeInProgress = False
                values = []

            elif gameData[0] == "Game closed!": # Game has been closed
                print("{}s game closed, saving and quitting program!".format(self.name))
                self.coord.request_stop()

            else:
                print("Invalid game data! got: {}".format(gameData[0]))