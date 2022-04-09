from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version



class A3C(object):
    def __init__(self, env, task, visualise):
        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step
                
                
"""
implement your code here
control synchronization and update between local and global networks
"""

