# Experimental main, for the new system with the game environment as a single object
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Queue
from threading import Thread
import os

import tensorflow as tf
import time

from A3CBootcampGame.FeedingGrounds.FeedingGrounds import FeedingGrounds
from A3CBootcampGame.ShootingGrounds.ShootingGrounds import ShootingGrounds
from ACNetwork import ACNetwork
from Worker import Worker
from init import Settings


def gameProcess(settings, gameDataQueue, playerActionQueue):
    game = ShootingGrounds(settings, gameDataQueue, playerActionQueue)
    while 1:
        game.runGameLoop()
        #time.sleep(settings.sleepTime)

def workerThread(worker, settings, sess, coord, saver):
    gameDataQueue, playerActionQueue = Queue(), Queue()
    game = Process(target=gameProcess, args=(settings, gameDataQueue, playerActionQueue))
    game.start()
    worker.work(settings, gameDataQueue, playerActionQueue, sess, coord, saver)
    game.terminate()

def main():

    settings = Settings()
    writer = tf.summary.FileWriter(settings.tbPath)
    globalNetwork = ACNetwork(settings, "global")
    workers = []
    for i in range(settings.workerCount):
        workers.append(Worker(settings, i))
    #writer.add_graph(tf.get_default_graph())
    #writer.flush()
    saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)

    workerThreads = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = settings.gpuMemoryFraction
    with tf.Session(config=config) as sess:
        if settings.loadCheckpoint:
            saver.restore(sess, settings.tfCheckpoint)
        else:
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        for worker in workers:
            thread = Thread(target=workerThread, args=(worker, settings, sess, coord, saver))
            thread.start()
            workerThreads.append(thread)
        coord.join(workerThreads)

if __name__ == "__main__":
    main()