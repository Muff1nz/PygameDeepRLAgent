
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process
import os

from ACNetworkLSTM import ACNetworkLSTM
from Trainer import Trainer
from init import Settings


def getClusterSpec(settings):
    port = 0
    ps = []
    for _ in range(settings.psCount):
        ps.append("localhost:{}".format(2000 + port))
        port += 1

    workers = []
    for _ in range(settings.trainerCount):
        workers.append("localhost:{}".format(2000 + port))
        port += 1

    clusterSpec = {"ps": ps,
                   "worker": workers}
    return clusterSpec

def process(number, task):
    import tensorflow as tf

    settings = Settings()
    clusterSpec = getClusterSpec(settings)
    cluster = tf.train.ClusterSpec(clusterSpec)


    print("Making a server")
    if task == "ps":
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        server = tf.train.Server(cluster,
                                 job_name=task,
                                 task_index=number,
                                 config=config)
        server.join()

    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        server = tf.train.Server(cluster,
                                 job_name=task,
                                 task_index=number,
                                 config=config)

    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = settings.gpuMemoryFraction

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:{}".format(number),
            cluster=clusterSpec)):
        print("MAKING MODEL")
        globalEpisodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        globalNetwork = ACNetworkLSTM(settings, "global")
        coord = tf.train.Coordinator()
        trainer = Trainer(settings, number, coord, globalEpisodes)
        #saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        initOp = tf.global_variables_initializer()

    monitor = tf.train.MonitoredTrainingSession(
        master=server.target,
        is_chief=(number == 0),
        config=config)


    with monitor as sess:
        print("MAKING A SESSION")
        trainer.init(sess)
        trainer.join()


def main():
    settings = Settings()
    trainers = []
    for i in range(settings.trainerCount):
        trainers.append(Process(target=process, args=(i, "worker")))

    for i in range(settings.psCount):
        trainers.append(Process(target=process, args=(i, "ps")))

    for trainer in trainers:
        trainer.start()

    for trainer in trainers:
        trainer.join()

if __name__ == "__main__":
    main()