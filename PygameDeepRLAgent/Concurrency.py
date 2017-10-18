
from multiprocessing import Process
from threading import Thread

class GameRunner(Process):
    def __init__(self, settings):
        Process.__init__(self)
        self.paqs = [] # playerActionQueues
        self.gdqs = [] # gameDataQueues
        self.settings = settings

    def addGame(self, gdq, paq):
        self.gdqs.append(gdq)
        self.paqs.append(paq)

    def run(self):
        import asyncio

        assert(len(self.gdqs) == len(self.paqs))

        game = self.settings.games[self.settings.game][0]
        games = []
        for gameData, playerAction in zip(self.gdqs, self.paqs):
            games.append(game(self.settings, gameData, playerAction))

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(*(game.run() for game in games)))
        # the below line wont really ever be reached, this process will just be terminated by the trainer when its done
        loop.close()

class TrainerRunner(Thread):
    def __init__(self, coord, queue):
        Thread.__init__(self, daemon=True)
        self.trainers = {}
        self.coord = coord
        self.trainerQueue = queue

    def addTrainer(self, trainer):
        self.trainers[trainer.number] = trainer

    def run(self):
        while not self.coord.should_stop():
            data = self.trainerQueue.get()
            self.trainers[data["trainer"]].train(data)

class WorkerRunner(Thread):
    def __init__(self, coord, gameDataQueue):
        Thread.__init__(self, daemon=True)
        self.workers = {}
        self.gameDataQueue = gameDataQueue
        self.coord = coord

    def addWorker(self, worker):
        self.workers[worker.number] = worker

    def run(self):
        while not self.coord.should_stop():
            data = self.gameDataQueue.get()
            self.workers[data[0]].work(data[1])


