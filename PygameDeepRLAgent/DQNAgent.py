import tensorflow as tf
import numpy as np
import random

from Actor import Actor

class DQNAgent(Actor):
    def __init__(self, settings, spritePath, sess, replayMemory, writer):
        super(DQNAgent, self).__init__(settings, spritePath)
        self.type = "player"
        self.lastAction = 0
        self.sess = sess
        self.rm = replayMemory

        self.dqnInput = tf.placeholder(dtype='float32', shape=[None, settings.stateDepth, settings.processedRes, settings.processedRes], name="dqnInput")
        self.dqn = self.cnn(self.dqnInput, settings)
        self.trainOp = self.buildTrainOp()
        self.mergedSummaries = self.summaries()
        self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)
        if settings.loadCheckpoint:
            self.saver.restore(self.sess, settings.tfGraphPath + "-" + str(settings.tfCheckpoint))
            print("Global step: " + str(self.step.eval()))
            print("Loading dqn from " + settings.tfGraphPath)

        self.writer = writer

    def update(self, ei, counter):
        dir = np.array([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])

        if not (counter % self.settings.deepRLRate): # Get a new action
            if random.random() > 0.01:
                action = self.forwardProp([self.rm.getState()])
                action = np.argmax(action)
            else:
                print("Acting off-policy!")
                action = random.randint(0, 24)
            self.lastAction = action
            if self.rm.full or self.rm.ei > self.settings.experienceMemorySizeStart:
                self.train(self.rm.getExperienceBatch(self.settings.experienceBatchSize))
        else:                                # Continue last action
            action = self.lastAction

        self.oldPos = self.pos.copy()
        self.pos += dir[action // 5] * self.speed
        d = dir[action % 5]
        if not (d[0] == 0 and d[1] == 0):
            self.ws.shoot(d, self.pos + int(self.size / 2), experienceIndex=ei)

        self.ws.update()

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
        self.ws.draw(screen)

    def reset(self):
        self.pos = np.array([self.settings.screenRes / 2, self.settings.screenRes / 5])
        self.ws.reset()

    def cnn(self, x, settings):
        x = tf.reshape(x, shape=[-1, settings.stateDepth, settings.processedRes, settings.processedRes, 1])
        print(x)

        with tf.name_scope('conv1'):
            self.wConv1 = tf.Variable(
                tf.truncated_normal([4, 8, 8, 1, 32], stddev=np.sqrt(2 / np.prod(x.get_shape().as_list()[1:])), name='wConv1'))
            self.bConv1 = tf.Variable(
                tf.truncated_normal([32], stddev=np.sqrt(2 / np.prod(x.get_shape().as_list()[1:])), name='bConv1'))

            conv1 = tf.nn.conv3d(x, self.wConv1, strides=[1, 4, 4, 4, 1], padding='SAME')
            conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.bConv1))
            print(conv1)
        with tf.name_scope('conv2'):
            self.wConv2 = tf.Variable(
                tf.truncated_normal([1, 4, 4, 32, 64], stddev=np.sqrt(2 / np.prod(conv1.get_shape().as_list()[1:])), name='wConv2'))
            self.bConv2 = tf.Variable(
                tf.truncated_normal([64], stddev=np.sqrt(2 / np.prod(conv1.get_shape().as_list()[1:])), name='bConv2'))

            conv2 = tf.nn.conv3d(conv1, self.wConv2, strides=[1, 1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.bConv2))
            print(conv2)
        with tf.name_scope('conv3'):
            self.wConv3 = tf.Variable(
                tf.truncated_normal([1, 2, 2, 64, 128], stddev=np.sqrt(2 / np.prod(conv2.get_shape().as_list()[1:])), name='wConv3'))
            self.bConv3 = tf.Variable(
                tf.truncated_normal([128], stddev=np.sqrt(2 / np.prod(conv2.get_shape().as_list()[1:])), name='bConv3'))

            conv3 = tf.nn.conv3d(conv2, self.wConv3, strides=[1, 1, 2, 2, 1], padding='SAME')
            conv3 = tf.nn.relu(tf.nn.bias_add(conv3, self.bConv3))
            print(conv3)
        with tf.name_scope('fc1'):
            self.wFc1 = tf.Variable(
                tf.truncated_normal([4 * 4 * 128, 512], stddev=np.sqrt(2 / np.prod(conv3.get_shape().as_list()[1:])), name="wFc1"))
            self.bFc1 = tf.Variable(
                tf.truncated_normal([512], stddev=np.sqrt(2 / np.prod(conv3.get_shape().as_list()[1:])), name='bFc1'))

            fc1 = tf.reshape(conv3, [-1, 4 * 4 * 128])
            fc1 = tf.matmul(fc1, self.wFc1)
            fc1 = tf.nn.relu(tf.nn.bias_add(fc1, self.bFc1))
        with tf.name_scope('fc2'):
            self.wFc2 = tf.Variable(
                tf.truncated_normal([512, 512], stddev=np.sqrt(2 / np.prod(fc1.get_shape().as_list()[1:])), name="wFc2"))
            self.bFc2 = tf.Variable(
                tf.truncated_normal([512], stddev=np.sqrt(2 / np.prod(fc1.get_shape().as_list()[1:])), name='bFc2'))

            fc2 = tf.matmul(fc1, self.wFc2)
            fc2 = tf.nn.relu(tf.nn.bias_add(fc2, self.bFc2))
        with tf.name_scope('out'):
            self.wOut = tf.Variable(
                tf.truncated_normal([512, 25], stddev=np.sqrt(2 / np.prod(fc2.get_shape().as_list()[1:])), name='wOut'))
            self.bOut = tf.Variable(
                tf.truncated_normal([25], stddev=np.sqrt(2 / np.prod(fc2.get_shape().as_list()[1:])), name='bOut'))

            output = tf.matmul(fc2, self.wOut)
            output = tf.nn.bias_add(output, self.bOut)
        return output

    def buildTrainOp(self):
        with tf.name_scope('trainer'):
            self.y = tf.placeholder(dtype='float32', shape=[None, 25])
            self.cost = tf.reduce_mean(tf.abs(tf.subtract(self.y, self.dqn)))
            self.step = tf.Variable(0, trainable=False, name='global_step')
            self.lr = tf.train.exponential_decay(0.00025, self.step, 50000, 0.94)
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)  # .minimize(self.cost, global_step=self.step)
            self.gradients, self.gradvars = zip(*self.optimizer.compute_gradients(self.cost))
            self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.5)
            trainOp = self.optimizer.apply_gradients(zip(self.gradients, self.gradvars), global_step=self.step)
            return trainOp

    def summaries(self):
        with tf.name_scope('cost_lr'):
            tf.summary.scalar('learningRate', self.lr)
            tf.summary.scalar('cost', self.cost)
            return tf.summary.merge_all()

    def forwardProp(self, state):
        return self.sess.run(self.dqn, feed_dict={self.dqnInput: state})

    def save(self):
        print("Saving model to " + self.settings.tfGraphPath)
        self.settings.tfCheckpoint = self.sess.run(self.step)
        self.saver.save(self.sess, self.settings.tfGraphPath, global_step=self.step)
        print("Saved model to " + self.settings.tfGraphPath)

    def train(self, batch):
        targets = []
        states = []

        maxQs = []
        for sample in batch:
            targets.append(sample.state)
            maxQs.append(sample.newState)

        targets = self.forwardProp(targets)
        maxQs = self.forwardProp(maxQs)
        self.reward = 0
        for i, sample in enumerate(batch):
            self.reward += sample.reward  # Just for logging, total reward in batch
            maxQ = maxQs[i].max()
            gamma = 0.99  # Placeholder
            if maxQ > 100:
                print("MaxQ: " + str(maxQ))
                print("MaxTarget: " + str(targets[i].max()))

            if sample.reward == -1: # Terminal state
                targets[i][sample.action] = sample.reward
            else:
                targets[i][sample.action] = sample.reward + maxQ * gamma

            states.append(sample.state)

        # Do fitting here
        summary, _ = self.sess.run([self.mergedSummaries, self.trainOp],
                                   feed_dict={
                                       self.dqnInput: states,
                                       self.y: targets
                                       }
                                   )
        self.writer.add_summary(summary, global_step=self.step.eval())



