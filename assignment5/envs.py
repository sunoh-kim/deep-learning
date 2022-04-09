import cv2
from gym.spaces.box import Box
import numpy as np
import gym
from gym import spaces
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

def create_env(env_id, client_id, remotes, **kwargs):
    """
    Implement your code here
    return "Pong-v0" environment
    """
    
"""
Implement your code here
that would help training such as preprocessing
"""

 