import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class ACNetworkLSTM:
    def __init__(self, settings, scope, step=0):
        with tf.variable_scope(scope):
            self.frame = tf.placeholder(shape=[None, settings.gameRes, settings.gameRes],
                                        dtype=tf.float32, name="frame")
            self.input = tf.reshape(self.frame, shape=[-1, settings.gameRes, settings.gameRes, 1])
            self.conv1 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=self.input,
                num_outputs=16,
                kernel_size=[16, 16],
                stride=[2, 2],
                scope="conv1"
            )
            self.conv2 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=self.conv1,
                num_outputs=32,
                kernel_size=[8, 8],
                stride=[2, 2],
                scope="conv2"
            )
            self.conv3 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=self.conv2,
                num_outputs=64,
                kernel_size=[4, 4],
                stride=[2, 2],
                scope="conv3"
            )

            lstmCell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            cInit = np.zeros(shape=(1, lstmCell.state_size.c), dtype=np.float32)
            hInit = np.zeros(shape=(1, lstmCell.state_size.h), dtype=np.float32)
            self.stateInit = [cInit, hInit]
            cIn = tf.placeholder(shape=[1, lstmCell.state_size.c], dtype=tf.float32, name="cIn")
            hIn = tf.placeholder(shape=[1, lstmCell.state_size.h], dtype=tf.float32, name="hIn")
            self.stateIn = [cIn, hIn]
            rnnIn = tf.expand_dims(self.flatten(self.conv3), [0])
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

            hidden = slim.fully_connected(rnnOut, 512, activation_fn=tf.nn.elu, scope="fc1")
            hidden2 = slim.fully_connected(hidden, 512, activation_fn=tf.nn.elu, scope="fc2")
            p0 = slim.fully_connected(hidden2, 256, activation_fn=tf.nn.elu, scope="p0")
            v0 = slim.fully_connected(hidden2, 256, activation_fn=tf.nn.elu, scope="v0")
            self.value = slim.fully_connected(v0, 1,
                                              activation_fn=None,
                                              weights_initializer=self.weightInit(1.0),
                                              scope="vOut")
            self.policy = slim.fully_connected(p0, settings.actionSize,
                                               activation_fn=None,
                                               weights_initializer=self.weightInit(0.01),
                                               scope="p1")

            with tf.variable_scope("pOut"):
                self.logits = tf.nn.softmax(self.policy)
                self.logLogits = tf.nn.log_softmax(self.policy)

            if scope != 'global':
                with tf.variable_scope("training"):
                    self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
                    self.actionsOnehot = tf.one_hot(self.actions, settings.actionSize, dtype=tf.float32)
                    self.targetV = tf.placeholder(shape=[None], dtype=tf.float32, name="targetV")
                    self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name="advantages")

                    # Loss functions
                    self.valueLoss = 0.5 * tf.reduce_sum(tf.square(self.targetV - tf.reshape(self.value, [-1])))
                    self.entropy = tf.reduce_sum(tf.multiply(self.logits, -self.logLogits))
                    self.policyLoss = -tf.reduce_sum(tf.reduce_sum(self.logLogits * self.actionsOnehot, [1]) * self.advantages)
                    self.loss = (settings.valueWeight * self.valueLoss +
                                 self.policyLoss -
                                 self.entropy * settings.entropyWeight)

                    # Get gradients from local network using local losses
                    localVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    self.gradients = tf.gradients(self.loss, localVars)
                    self.varNorms = tf.global_norm(localVars)
                    self.grads, self.gradNorms = tf.clip_by_global_norm(self.gradients, 40)

                    self.lr = tf.train.exponential_decay(settings.learningRate,
                                                         step,
                                                         settings.lrDecayStep,
                                                         settings.lrDecayRate)
                    optimizer = tf.train.AdamOptimizer(self.lr);

                    # Apply local gradients to global network
                    globalVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                    self.applyGradsGlobal = optimizer.apply_gradients(zip(self.grads, globalVars), global_step=step)

    def weightInit(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

    def flatten(self, x):
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
