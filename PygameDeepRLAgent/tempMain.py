# Experimental main, for the new system with the game environment as a single object
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Queue
from threading import Thread
import tensorflow as tf

from ACNetwork import ACNetwork
from Worker import Worker
from init import Settings
from ClusterCube import ClusterCube # The actual game

def gameProcess(settings, gameDataQueue, playerActionQueue):
    game = ClusterCube(settings, gameDataQueue, playerActionQueue)
    while 1:
        game.runGameLoop()

def workerThread(worker, settings, sess, coord):
    gameDataQueue, playerActionQueue = Queue(), Queue()
    game = Process(target=gameProcess, args=(settings, gameDataQueue, playerActionQueue))
    game.start()
    while 1:
        worker.work(settings, gameDataQueue, playerActionQueue, sess, coord)

def main():
    settings = Settings()

    trainer = tf.train.AdamOptimizer()
    globalNetwork = ACNetwork(settings, "global")
    workers = []
    for i in range(6):
        workers.append(Worker(settings, i, trainer))

    workerThreads = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        for worker in workers:
            thread = Thread(target=workerThread, args=(worker, settings, sess, coord))
            thread.start()
            workerThreads.append(thread)
        coord.join(workerThreads)


if __name__ == "__main__":
    main()