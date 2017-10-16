from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process

from init import Settings

def utilityThread(settings, sess, saver, globalEpisodes, coord):
    import time
    lastSave = sess.run(globalEpisodes)
    while not coord.should_stop():
        episodeNumber = sess.run(globalEpisodes)
        print("|| __ {} __ || Global episodes: {} ||".format(settings.activity, sess.run(globalEpisodes)))

        if (episodeNumber > 5000 + lastSave and settings.saveCheckpoint):
            print("UtilityThread is saving the model!")
            saver.save(sess, settings.tfGraphPath, episodeNumber)
            lastSave = episodeNumber

        if (episodeNumber > settings.trainingEpisodes):
            coord.request_stop()

        time.sleep(2) # This function needs little CPU time

    if (settings.saveCheckpoint):
        print("Program is terminating, utilityThread is saving the model!")
        saver.save(sess, settings.tfGraphPath, sess.run(globalEpisodes))

def run(settings = Settings()):
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

        for thread in threads:
            thread.start()
        utilityThread(settings, sess, saver, globalEpisodes, coord)
        coord.join(threads)

def startProcess(settings, lr):
    settings.learningRate = lr
    print("Learning rate: " + str(lr))
    settings.generateActivity()
    settings.generatePaths()
    process = Process(target=run, args=(settings,))
    process.start()
    return process

def join(processes):
    for process in processes:
        process.join()

def lrSweep(): # This function will test varius learning rates
    exp = 4
    lr = 1
    for i in range(6):
        p = startProcess(lr / (10 ** exp))
        p.join()
        lr = 1 if exp == 5 else 5
        if (lr == 5):
            exp += 1

def main():
    processes = []
    #conf1 = Settings()
    #processes.append(startProcess(conf1, 1e-4))
    conf2 = Settings()
    conf2.tfCheckpoint = 'C:\deepRLAgent\Agent\\5e-05LR_0.98LRDR_140LRDS_4DLRRate_16T-2W_100000Episodes\-70868'
    processes.append(startProcess(conf2, 5e-5))
    join(processes)



if __name__ == "__main__":
    main()