from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread
import os

import tensorflow as tf

from ACNetwork import ACNetwork
from Trainer import Trainer
from init import Settings


def utilityThread(settings, sess, saver, globalEpisodes, coord):
    lastEpisodePrint = 0
    lastSave = 0
    while not coord.should_stop():
        episodeNumber = sess.run(globalEpisodes)
        if (episodeNumber % 5 == 0 and episodeNumber != lastEpisodePrint):
            print("Global episodes: {}".format(sess.run(globalEpisodes)))
            lastEpisodePrint = episodeNumber

        if (episodeNumber % 2000 == 0 and episodeNumber != lastSave):
            print("UtilityThread is saving the model!")
            saver.save(sess, settings.tfGraphPath + settings.agentName, episodeNumber)
            lastSave = episodeNumber

        if (episodeNumber > settings.trainingEpisodes):
            coord.request_stop()

    print("Program is terminating, utilityThread is saving the model!")
    saver.save(sess, settings.tfGraphPath + settings.agentName, sess.run(globalEpisodes))

def main():
    settings = Settings()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = settings.gpuMemoryFraction
    with tf.Session(config=config) as sess:
        globalEpisodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        globalNetwork = ACNetwork(settings, "global")
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

if __name__ == "__main__":
    main()