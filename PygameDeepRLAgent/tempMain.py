# Experimental main, for the new system with the game environment as a single object

from init import Settings
from ClusterCube import ClusterCube # The actual game
from multiprocessing import Process

def gameProcess(settings):
    game = ClusterCube(settings)
    while 1:
        game.runGameLoop()

def main():
    settings = Settings()
    workers = []
    for _ in range(4):
        worker = Process(target=gameProcess, args=(settings,))
        worker.start()
        workers.append(worker)

if __name__ == "__main__":
    main()