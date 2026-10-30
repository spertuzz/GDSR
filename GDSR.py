"""
GEOMETRY DASH MODEL
"""

import matplotlib.pyplot as plt
from IPython.display import clear_output
from mathstuff import *
from nets import *


# GD Ship Regression Model
class GD_ship_regressor:
  """
  x_values = []     # Set of real x values for training
  y_values = []     # Set of real y values for training (corresponding in order to x values)
  gen = 100         # Amount of networks per generation
  top = 10          # Amount of selected candidates per generation
  resolution = 100  # Amount of frames in the model's path
  scroll = 1        # Horizontal scrolling speed
  grav = 1          # Vertical gravity
  thrust = 1        # Vertical upthrust force
  margin = 0.5      # Extra room given to the bot so it can, like, get into position before the curve
  """

  def __init__(self, x_values, y_values, gen=100, top=10, resolution=100, scroll=1, grav=1, thrust=1, margin=0.4, y_margin=0):
    # Set basic values
    self.x_values = x_values
    self.y_values = y_values

    self.scroll = scroll
    self.grav = grav
    self.thrust = thrust
    self.margin = margin
    self.y_margin = y_margin
    self.resolution = resolution

    # Create neuroevolution model
    self.model = evolution(
        wither_rate=0.995, max_wither=0.01, gen=gen, top=top, score_func=self.score,
        rand=uniform_rand, b_rand=uniform_rand,
        input_count=5, input_act=lin,
        hidden_layers=4, hidden_neurons=32, hidden_act=ReLU,
        output_count=1, output_act=tanh
    )

  def simulate(self, network):
    x_res = []
    y_res = []

    # Generate important movement variables
    length = max(self.x_values) - min(self.x_values)                      # Length of the system in x units
    height = max(self.y_values) - min(self.y_values) + 2 * self.y_margin  # Length of the system in y units
    dt = 5 / self.resolution                                              # Time per frame (whole path is taken at 5 seconds)
    h_speed = self.scroll * length / 5                                    # Horizontal speed
    g_force = 2 * height * self.grav                                      # Downforce
    up_force = 2 * height * self.thrust                                   # Upforce
    end_dist = max(self.x_values)                                         # Final point in the path

    # Create concurrent moving variables
    y_vel = 0
    y_pos = (max(self.y_values) + min(self.y_values)) / 2
    x_pos = min(self.x_values) - length * self.margin

    # Start movement loop
    while True:
      # Find the next target obstacle's coordinates to feed into the network
      next_x = min(self.x_values) - max(abs(min(self.x_values)), abs(max(self.x_values))) * 100
      next_y = min(self.y_values) - max(abs(min(self.y_values)), abs(max(self.y_values))) * 100
      for i in range(len(self.x_values)):
        # Find next target
        if x_pos < self.x_values[i]:
          next_x = self.x_values[i]
          next_y = self.y_values[i]
          break

      # Fire neural network
      input_arr = [x_pos, y_pos, y_vel, next_x, next_y]
      click = network.fire(input_arr)[0] > 0  # Only click if output is positive

      if click:  # Add thrust if the bot clicked
        y_vel += dt * up_force
        if y_vel > up_force: y_vel = up_force
      else:  # Let it fall if it didn't click
        y_vel -= dt * g_force
        if y_vel < -g_force: y_vel = -g_force

      # Update movement variables
      x_pos += h_speed * dt
      y_pos += y_vel * dt

      # Append result to path
      x_res.append(x_pos)
      y_res.append(y_pos)

      if x_pos > end_dist:  # Check if the path is done
        break

    return (x_res, y_res)  # Return resulting path

  def get_result(self, input, x_res, y_res):
    if input < x_res[0] or input > x_res[-1]:
      raise ValueError("Input must be inside the model's margins.")

    for i in range(len(x_res)):
      # Find range between frames where the value lives
      if input <= x_res[i]:
        if i == len(x_res) - 1:
          return y_res[-1]  # Send last value if it lands on the final frame
        x1 = x_res[i]
        x2 = x_res[i + 1]
        y1 = y_res[i]
        y2 = y_res[i + 1]
        return ((input - x1) / (x2 - x1)) * (y2 - y1) + y1  # Return linearly interpolated value between both closest frames

  def fire(self, input, network=None):
    if network is None:
      network = self.model.net

    # Get model result on a certain input
    results = self.simulate(network=network)
    x_res = results[0]
    y_res = results[1]
    return self.get_result(input, x_res, y_res)

  def score(self, networks):
    tuples = []
    for n in networks:
      error = 0
      results = self.simulate(network=n)  # Simulate the network and store results for efficiency
      x_res = results[0]
      y_res = results[1]
      for i in range(len(self.x_values)):
        y_real = self.y_values[i]
        y_pred = self.get_result(self.x_values[i], x_res, y_res)  # Score model based on SSE (Sum of Squared Errors)
        error += (y_pred - y_real) ** 2
      tuples.append((n, -error))  # Add model-error pairs

    return tuples

  def visualize(self, model, kill_old=False):
    clear_output()

    # Gather simulated results (path)
    results = self.simulate(network=model.net)
    x_res = results[0]
    y_res = results[1]
    if kill_old:
      results_c = self.simulate(network=model.candidates[0])
      x_res_c = results_c[0]
      y_res_c = results_c[1]

    # Graph both the real and predicted paths
    plt.figure(figsize=(10, 8))
    plt.ylim(min(self.y_values) - self.y_margin, max(self.y_values) + self.y_margin)
    plt.xlim(min(self.x_values), max(self.x_values))
    plt.plot(self.x_values, self.y_values, label='True Curve', color='blue')
    if kill_old: plt.plot(x_res_c, y_res_c, label='Current Generation Best', color='green')
    plt.plot(x_res, y_res, label='All-Time Best', color='orange')
    plt.title('Generation ' + str(self.model.epochs) + ' - Tracing the curve sqrt(x) * sin(1/2 * x * e^(-x/20))')
    plt.legend()
    plt.tight_layout()
    plt.show()

  def train(self, gens=500, kill_old=False, cross_mode='pairwise', gamma_mode='complete', min_gamma=0, max_gamma=1, gamma_exp=0.01, debug=True, visualize=False):
    vis = None
    if visualize:  # Determine whether the visualizer is to be used
      vis = self.visualize
    self.model.train(  # Train neuroevolution model
        gens=gens, kill_old=kill_old, cross_mode=cross_mode,
        gamma_mode=gamma_mode, min_gamma=min_gamma, max_gamma=max_gamma, gamma_exp=gamma_exp,
        debug=debug, visualizer=vis
    )