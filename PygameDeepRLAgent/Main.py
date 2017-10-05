from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process

from init import Settings

def utilityThread(settings, sess, saver, globalEpisodes, coord):
    lastEpisodePrint = 0
    lastSave = 0
    while not coord.should_stop():
        episodeNumber = sess.run(globalEpisodes)
        if (episodeNumber % 5 == 0 and episodeNumber != lastEpisodePrint):
            print("Global episodes: {}".format(sess.run(globalEpisodes)))
            lastEpisodePrint = episodeNumber

        if (episodeNumber % 10000 == 0 and episodeNumber != lastSave):
            print("UtilityThread is saving the model!")
            saver.save(sess, settings.tfGraphPath + settings.agentName, episodeNumber)
            lastSave = episodeNumber

        if (episodeNumber > settings.trainingEpisodes):
            coord.request_stop()

    print("Program is terminating, utilityThread is saving the model!")
    saver.save(sess, settings.tfGraphPath + settings.agentName, sess.run(globalEpisodes))

def run(settings = Settings()):
    from threading import Thread
    import tensorflow as tf
    import os

    from ACNetworkLSTM import ACNetworkLSTM
    from Trainer import Trainer

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.device("/cpu:0"):
            globalEpisodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        globalNetwork = ACNetworkLSTM(settings, "global")
        coord = tf.train.Coordinator()
        threads = []
        for i in range(settings.trainerCount):
            threads.append(Trainer(settings, sess, i, coord, globalEpisodes))
        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)

        if settings.loadCheckpoint:
            saver.restore(sess, settings.tfCheckpoint)
        else:
            sess.run(tf.global_variables_initializer())

        threads.append(Thread(target=utilityThread, args=(settings, sess, saver, globalEpisodes, coord)))
        for thread in threads:
            thread.start()
        coord.join(threads)

def startProcess(exp):
    settings = Settings()
    settings.learningRate = 1 / (10 ** exp)
    print("Learning rate: " + str(settings.learningRate))
    settings.generateActivity()
    settings.generatePaths()
    process = Process(target=run, args=(settings,))
    process.start()
    return process

def join(processes):
    for process in processes:
        process.join()

def main():
    exp = 0
    processes = []
    for i in range(5):
        exp += 1
        processes.append(startProcess(exp))
        exp += 1
        processes.append(startProcess(exp))
        join(processes)
        processes = []



if __name__ == "__main__":
    main()