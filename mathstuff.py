"""
ACTIVATION FUNCTIONS
"""

import math
import random

# For input layers mainly
def lin(x):
  return x

# Standard for hidden layers
def ReLU(x):
  return max(0, x)

def leakyReLU(x):
  return max(0.01 * x, x)

# Standard for output layers
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def tanh(x):
  return math.tanh(x)

"""
RANDOMS
"""

# Common randoms
constant_rand = {'type': 'constant', 'c': 0}             # Returns a constant 'c' always
uniform_rand = {'type': 'uniform', 'min': -1, 'max': 1}  # Returns a random value between 'min' and 'max' uniformly
normal_rand = {'type': 'normal', 'mean': 0, 'std': 1}    # Returns a random value from a normal distribution with said parameters

def create_rand(r):
  if r is None:
    return 0

  t = r['type']
  if t == 'constant':
    return r['c']
  elif t == 'uniform':
    return random.uniform(r['min'], r['max'])
  elif t == 'normal':
    return random.normalvariate(r['mean'], r['std'])

  raise ValueError('Expected a valid rand type.')