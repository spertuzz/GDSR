"""
DEMO (ish)
"""

import matplotlib.pyplot as plt
from mathstuff import *
from nets import *
from GDSR import *

# Create sample points for model to fly through
def path(x):
  # return 1
  return (x ** 0.5) * math.sin(0.5 * x * math.exp(-x/20))

# Generate x, y arrays
x_values = []
y_values = []
for i in range(0, 301):
  n = (30 - 0) * (i / 300)
  x_values.append(n)
  y_values.append(path(n))

# Start training model and test
GD_model = GD_ship_regressor(x_values, y_values, gen=100, top=10, resolution=300, scroll=1, grav=1, thrust=1, margin=0.4, y_margin=1)
GD_model.train(gens=500, kill_old=True, visualize=True)