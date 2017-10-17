import numpy as np

# The worker class is a member of the trainer class, the trainer can have multiple workers
class Worker():
    def __init__(self, settings, sess, trainerName, number, network, queues, coord):
        self.localAC = network
        self.name = '{}_worker{}'.format(trainerName, number)
        self.number = number
        self.settings = settings
        self.trainerQueue = queues["trainer"]
        self.coord = coord
        self.sess = sess
        self.gameDataQueue = queues["gameData"]
        self.playerActionQueue = queues["playerAction"]
        self.game = None
        self.playerActionQueue.put(["WindowSettings", True if self.name == "trainer0_worker0" else False])

    def work(self):
        if self.gameDataQueue.empty():
            return # Nothing to do here, return control to caller
        gameData = self.gameDataQueue.get() # Get data from the game
        if gameData[0] == "CurrentFrame": # Process the next action based on frame
            frame = gameData[1]
            feedDict = {self.localAC.frame: [frame],
                        self.localAC.stateIn[0]: self.rnnState[0],
                        self.localAC.stateIn[1]: self.rnnState[1]}
            actionDist, value, self.rnnState = self.sess.run([self.localAC.logits,
                                                         self.localAC.value,
                                                         self.localAC.stateOut],
                                                         feed_dict=feedDict)
            action = np.random.choice(actionDist[0], p=actionDist[0])
            action = np.argmax(actionDist==action)

            self.playerActionQueue.put(action)
            self.values.append(value[0,0])

        elif gameData[0] == "Bootstrap": # Bootstrap from bootstrap data
            print("{} is bootstrapping!".format(self.name))
            bootstrapValues = self.values[0:self.settings.bootStrapCutOff]
            values = self.values[self.settings.bootStrapCutOff::]
            episodeData = gameData[1]
            workerData = {"episodeData": episodeData,
                          "values": bootstrapValues,
                          "bootStrapValue": values[0],
                          "score": -1,
                          "worker": self.number}
            self.trainerQueue.put(workerData)

        elif gameData[0] == "EpisodeData": # Episode is finished, perform training and logging
            episodeData = gameData[1]
            score = self.gameDataQueue.get()[1]
            workerData = {"episodeData": episodeData,
                          "values": self.values,
                          "bootStrapValue": 0,
                          "score": score,
                          "worker": self.number}
            self.trainerQueue.put(workerData)
            self.values = []
            self.rnnState = self.localAC.stateInit

        elif gameData[0] == "Game closed!": # Game has been closed
            print("{}s game closed, saving and quitting program!".format(self.name))
            self.coord.request_stop()

        else:
            print("Invalid game data! got: {}".format(gameData[0]))
