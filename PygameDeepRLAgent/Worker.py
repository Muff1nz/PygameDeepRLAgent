import numpy as np

# The worker class is a member of the trainer class, the trainer can have multiple workers
class Worker():
    def __init__(self, settings, sess, number, trainerNumber, network, queues, coord):
        self.localAC = network
        self.name = 'worker{}'.format(number)
        self.number = number
        self.settings = settings
        self.trainerQueue = queues["trainer"]
        self.trainerNumber = trainerNumber
        self.coord = coord
        self.sess = sess
        self.playerActionQueue = queues["playerAction"]
        self.game = None
        self.playerActionQueue.put({"WindowSettings": True if self.name == "worker0" else False,
                                    "worker": number})

        self.episodeInProgress = True
        self.values = []

    def work(self, gameData):
        gameData # Get data from the game
        if gameData[0] == "CurrentFrame": # Process the next action based on frame
            frame = gameData[1]
            feedDict = {self.localAC.frame: [frame]}
            actionDist, value = self.sess.run([self.localAC.logits,
                                               self.localAC.value],
                                               feed_dict=feedDict)
            action = np.random.choice(actionDist[0], p=actionDist[0])
            action = np.argmax(actionDist==action)

            self.playerActionQueue.put(action)
            self.values.append(value[0,0])

        elif gameData[0] == "Bootstrap": # Bootstrap from bootstrap data
            print("{} is bootstrapping!".format(self.name))
            episodeData = gameData[1]
            bootstrapValues = self.values[0:len(episodeData)]
            self.values = self.values[len(episodeData)::]
            workerData = {"episodeData": episodeData,
                          "values": bootstrapValues,
                          "bootStrapValue": self.values[0],
                          "score": 0,
                          "worker": self.number,
                          "trainer": self.trainerNumber}
            self.trainerQueue.put(workerData)

        elif gameData[0] == "EpisodeData": # Episode is finished, perform training and logging
            episodeData = gameData[1]
            score = gameData[2]
            workerData = {"episodeData": episodeData,
                          "values": self.values,
                          "bootStrapValue": 0,
                          "score": score,
                          "worker": self.number,
                          "trainer": self.trainerNumber}
            self.trainerQueue.put(workerData)
            self.values = []

        elif gameData[0] == "Game closed!": # Game has been closed
            print("{}s game closed, saving and quitting program!".format(self.name))
            self.coord.request_stop()

        else:
            print("Invalid game data! got: {}".format(gameData[0]))