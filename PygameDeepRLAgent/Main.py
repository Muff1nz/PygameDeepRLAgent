# Experimental main, for the new system with the game environment as a single object
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Queue
from threading import Thread
import tensorflow as tf
import pygame
import time

from ACNetwork import ACNetwork
from Worker import Worker
from init import Settings
from ClusterCube import ClusterCube # The actual game

def gameProcess(settings, gameDataQueue, playerActionQueue):
    game = ClusterCube(settings, gameDataQueue, playerActionQueue)
    while 1:
        time.sleep(settings.sleepTime)
        game.runGameLoop()

def workerThread(worker, settings, sess, coord, saver):
    gameDataQueue, playerActionQueue = Queue(), Queue()
    game = Process(target=gameProcess, args=(settings, gameDataQueue, playerActionQueue))
    game.start()
    worker.work(settings, gameDataQueue, playerActionQueue, sess, coord, saver)
    game.terminate()

def main():
    settings = Settings()

    globalEpisodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer()
    globalNetwork = ACNetwork(settings, "global")
    workers = []
    for i in range(settings.workerCount):
        workers.append(Worker(settings, i, trainer, globalEpisodes))
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    workerThreads = []
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = settings.gpuMemoryFraction
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        for worker in workers:
            thread = Thread(target=workerThread, args=(worker, settings, sess, coord, saver))
            thread.start()
            workerThreads.append(thread)
        coord.join(workerThreads)

if __name__ == "__main__":
    main()