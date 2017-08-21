import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class ACNetwork:
    def __init__(self, settings, scope, step=0):
        with tf.variable_scope(scope):
            self.processedFrame = tf.placeholder(shape=[None, settings.processedRes, settings.processedRes],
                                                 dtype=tf.float32, name="processedFrame")
            self.input = tf.reshape(self.processedFrame, shape=[-1, settings.processedRes, settings.processedRes, 1])
            self.conv1 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=self.input,
                num_outputs=16,
                kernel_size=[8, 8],
                stride=[4, 4],
                padding='VALID'
            )
            self.conv2 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=self.conv1,
                num_outputs=32,
                kernel_size=[4, 4],
                stride=[2, 2],
                padding='VALID'
            )
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256,
                                          activation_fn=tf.nn.elu,
                                          weights_initializer=self.weightInit(0.01),
                                          biases_initializer=tf.constant_initializer(0)
            )
            '''
            lstmCell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            cInit = np.zeros(shape=(1, lstmCell.state_size.c), dtype=np.float32)
            hInit = np.zeros(shape=(1, lstmCell.state_size.h), dtype=np.float32)
            self.stateInit = [cInit, hInit]
            cIn = tf.placeholder(shape=[1, lstmCell.state_size.c], dtype=tf.float32, name="cIn")
            hIn = tf.placeholder(shape=[1, lstmCell.state_size.h], dtype=tf.float32, name="hIn")
            self.stateIn = [cIn, hIn]
            rnnIn = tf.expand_dims(hidden, [0])
            stepSize = tf.shape(self.input)[:1]
            stateIn = tf.contrib.rnn.LSTMStateTuple(cIn, hIn)
            lstmOutputs, lstmState = tf.nn.dynamic_rnn(
                lstmCell,
                rnnIn,
                initial_state=stateIn,
                sequence_length=stepSize,
                time_major=False
            )
            lstmC, lstmH = lstmState
            self.stateOut = [lstmC[:1, :], lstmH[:1, :]]
            rnnOut = tf.reshape(lstmOutputs, [-1, 256])
            '''
            self.policy = slim.fully_connected(
                hidden, settings.actionSize,
                activation_fn=None,
                weights_initializer=self.weightInit(0.01),
                biases_initializer=tf.constant_initializer(0)
            )
            self.value = slim.fully_connected(
                hidden, 1,
                activation_fn=None,
                weights_initializer=self.weightInit(1.0),
                biases_initializer=tf.constant_initializer(0)
            )

            self.logits = tf.nn.softmax(self.policy)
            self.logLogits = tf.nn.log_softmax(self.policy)

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
                self.actionsOnehot = tf.one_hot(self.actions, settings.actionSize, dtype=tf.float32)
                self.targetV = tf.placeholder(shape=[None], dtype=tf.float32, name="targetV")
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name="advantages")

                # Loss functions
                self.valueLoss = 0.5 * tf.reduce_sum(tf.square(self.targetV - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.logits * self.logLogits)
                self.policyLoss = -tf.reduce_sum(tf.reduce_sum(self.logLogits * self.actionsOnehot, [1]) * self.advantages)
                self.loss = 0.5 * self.valueLoss + self.policyLoss - self.entropy * 0.01

                # Get gradients from local network using local losses
                localVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, localVars)
                self.varNorms = tf.global_norm(localVars)
                grads, self.gradNorms = tf.clip_by_global_norm(self.gradients, 40)

                optimizer = tf.train.AdamOptimizer(1e-4)

                # Apply local gradients to global network
                globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.applyGradsGlobal = optimizer.apply_gradients(zip(grads, globalVars), global_step=step)

    def weightInit(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer
