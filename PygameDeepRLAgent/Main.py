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
            print("Training done is done!!!!")
            coord.request_stop()

        time.sleep(2) # This function needs little CPU time

    if (settings.saveCheckpoint):
        print("Program is terminating, saving the model!")
        saver.save(sess, settings.tfGraphPath, sess.run(globalEpisodes))

def run(settings = Settings()):
    import tensorflow as tf
    import os

    from ACNetwork import ACNetwork

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 1.0
    with tf.Session() as sess:
        globalEpisodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        globalNetwork = ACNetwork(settings, "global")
        coord = tf.train.Coordinator()

        trainers, workers, games = setupConcurrency(settings, sess, coord, globalEpisodes)

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)

        if settings.loadCheckpoint:
            saver.restore(sess, settings.tfCheckpoint)
        else:
            sess.run(tf.global_variables_initializer())

        for thread in trainers:
            thread.start()
        for thread in workers:
            thread.start()
        for process in games:
            process.start()

        utilityThread(settings, sess, saver, globalEpisodes, coord)

        for game in games:
            game.terminate()

def setupConcurrency(settings, sess, coord, globalEpisodes):
    from queue import Queue
    from multiprocessing import Queue as PQueue

    import Concurrency
    from Worker import Worker
    from Trainer import Trainer

    trainingQueues = []
    trainerThreads = []
    for i in range(settings.trainerThreads):
        queue = Queue(100)
        trainingQueues.append(queue)
        trainerThreads.append(Concurrency.TrainerRunner(coord, queue))

    gameDataQueues = []
    workerThreads = []
    for i in range(settings.workerThreads):
        gameDataQueue = PQueue(100)
        gameDataQueues.append(gameDataQueue)
        workerThreads.append(Concurrency.WorkerRunner(coord, gameDataQueue))

    gameProcesses = []
    for i in range(settings.gameProcesses):
        gameProcesses.append(Concurrency.GameRunner(settings))


    trainers = []
    for i in range(settings.trainers):
        trainer = Trainer(settings, sess, i, coord, globalEpisodes)
        trainers.append(trainer)
        trainerThreads[i % len(trainerThreads)].addTrainer(trainer)

    for i in range(settings.workers):
        playerActionQueue = PQueue(100)

        queues = {"trainer": trainingQueues[i%len(trainingQueues)],
                  "gameData": gameDataQueue,
                  "playerAction": playerActionQueue}
        trainer = trainers[i%len(trainers)]
        worker = Worker(settings, sess, i, trainer.number, trainer.localAC, queues, coord)
        workerThreads[i % len(workerThreads)].addWorker(worker)
        gameProcesses[i % len(gameProcesses)].addGame(gameDataQueues[i % len(gameDataQueues)], playerActionQueue)

    return trainerThreads, workerThreads, gameProcesses



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
    conf2.tfCheckpoint = 'C:\deepRLAgent\Agent\\1.37\\5e-05LR_0.98LRDR_150LRDS_4DLRRate_16T-32W_200000Episodes\-50529'
    processes.append(startProcess(conf2, 1e-5))
    join(processes)

if __name__ == "__main__":
    main()