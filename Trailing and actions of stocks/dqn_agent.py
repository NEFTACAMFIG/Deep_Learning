%cd '/content/drive/MyDrive/Mis Documentos/Curso de Python/Deep Reinforcement Learning. Project'
%ls

pip install -q keras-rl2
pip install -q tf-agents[reverb]

from deng_env import Deng as DengEnv
from trail_env import Trail as TrailEnv

from keras import __version__
# Keras modules
import tensorflow as tf
tf.keras.__version__ = __version__

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, PReLU
from keras.optimizers import Adam, legacy
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session

# Keras - rl modules
try:
  from rl.agents.dqn import DQNAgent
except:
  tf.keras.__version__ = __version__
  from rl.agents.dqn import DQNAgent

from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Miscellaneous imports
import plotter as plt
import numpy as np
import os
from pathlib import Path
import datetime
import json

ENV_NAME = 'trading-rl'

trailing = 'trailing'
deng = 'deng'

METHOD = trailing  # Choose between environments

directory = str(Path.cwd().parent)  # Get the parent directory of the current working directory
data_directory = directory + "/data"

# Hardware Parameters
CPU = True  # Selection between CPU or GPU
CPU_cores = 7  # If CPU, how many cores
GPU_mem_use = 0.25  # In both cases the GPU mem is going to be used, choose fraction to use

train_data = data_directory + '/train_data.npy'  # path to training data
MAX_DATA_SIZE = 12000  # Maximum size of data
DATA_SIZE = MAX_DATA_SIZE  # Size of data you want to use for training

test_data = data_directory + '/test_data.npy'  # path to test data
TEST_EPOCHS = 1  # How many test runs / epochs
TEST_POINTS = [0]  # From which point in the time series to start in each epoch
TEST_STEPS = 2000  # For how many points to run the epoch
