import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        """
        Implement your code here
        """
        
    def act(self, ob, c, h):
        """
        Implement your code here
        """

    def value(self, ob, c, h):
        """
        Implement your code here
        """

