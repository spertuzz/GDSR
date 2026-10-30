"""
NEURAL NETWORKS
"""

from mathstuff import *
import math

# Neuron/node class for neural networks
class neuron:
  """
  w = None      # Weights
  b = None      # Bias
  act = None    # Activation function
  """

  def __init__(self, act=lin, w=None, b=0):
    self.act = act
    self.w = w
    if w is None: self.w = []
    self.b = b

  def fire(self, input):
    if len(input) != len(self.w):
      raise ValueError('Number of inputs must be equal to the number of weights.')

    # Calculate the output for this node
    output = self.b
    for i in range(len(self.w)):
      output += self.w[i] * input[i]

    return self.act(output)

  def copy(self):
    return neuron(act=self.act, w=self.w.copy(), b=self.b)


# Neural network layer class
class layer:
  """
  input = False   # Whether or not it's an input layer
  neurons = None  # Neurons in this layer
  next = None     # Next layer after this one
  """

  def __init__(self, neurons=[], next=None, input=False):
    self.neurons = neurons
    self.next = next
    self.input = input

  def fire(self, input):
    output = []

    if not self.input:  # If it's not an input give every neuron every input
      for n in self.neurons:
        output.append(n.fire(input))
    else:  # If it's an input node, give each neuron its corresponding input
      for i in range(len(self.neurons)):
        output.append(self.neurons[i].fire([input[i]]))

    if self.next is None:  # If it's the last layer, send the final output back
      return output
    else:  # Send the output down the chain
      return self.next.fire(output)

  def copy(self):
    new_n = []
    for n in self.neurons:
      new_n.append(n.copy())

    return layer(neurons=new_n, next=None, input=self.input)


# Neural network class
class network:
  """
  layers = None  # Layers in this network, ordered as 'layers[0]' being input and 'layers[-1]' being output
  """

  def __init__(
        self, layers=None,
        rand=uniform_rand, b_rand=None,                       # Type of random distribution used
        input_count=1, input_act=lin,                         # Input layer configs
        hidden_layers=3, hidden_neurons=10, hidden_act=ReLU,  # Hidden layer configs
        output_count=1, output_act=sigmoid                    # Output layer configs
    ):
    # Check if layers are already defined before randomizing
    if layers is not None:
      self.layers = layers
    else:
      # Create random number generating functions to be called for all future neurons
      call_rand = lambda: create_rand(rand)
      call_rand_b = lambda: create_rand(b_rand)

      # Build all layers backwards (it's easier to tie a layer to the next this way)
      layers = []

      # Output layer
      output_n = []
      for i in range(output_count):
        w = []
        for j in range(hidden_neurons):
          w.append(call_rand())
        output_n.append(neuron(act=output_act, w=w, b=call_rand_b()))
      output_layer = layer(input=False, neurons=output_n, next=None)
      layers.append(output_layer)

      # Hidden layers
      for i in range(hidden_layers):
        hidden_n = []
        if i == hidden_layers - 1:
          max = input_count
        else:
          max = hidden_neurons

        for j in range(hidden_neurons):
          w = []
          for k in range(max):
            w.append(call_rand())
          hidden_n.append(neuron(act=hidden_act, w=w, b=call_rand_b()))

        hidden_layer = layer(input=False, neurons=hidden_n, next=layers[-1])
        layers.append(hidden_layer)

      # Input layer
      input_n = []
      for i in range(input_count):
        input_n.append(neuron(act=input_act, w=[1], b=0))
      input_layer = layer(input=True, neurons=input_n, next=layers[-1])
      layers.append(input_layer)

      # Reverse layers to lock layers[0] as the input layer
      layers.reverse()
      self.layers = layers

  def fire(self, input):
    return self.layers[0].fire(input)  # Evaluate at 'input'

  def copy(self):
    new_layers = []
    prev = None
    for l in reversed(self.layers):
      new_l = l.copy()
      new_l.next = prev
      new_layers.append(new_l)
      prev = new_l
    new_layers.reverse()

    return network(layers=new_layers)

"""
NEUROEVOLUTION
"""

# Evolving neural network model
class evolution:
  """
  net = None          # Current best model
  base_net = None     # Function to create the default initial network
  candidates = []     # Current top scoring candidate models
  mu = 0.99           # 'Withering rate': rate at which random mutations dissipate
  score_func = None   # Function to score networks
  gen = 10000         # Size of a generation
  top = 100           # Amount of top performers to be conserved
  call_rand = None    # Function for creating random distributions for weights
  call_rand_b = None  # Function for creating random distributions for biases
  epochs = 0          # Current generation epoch
  """

  def __init__(
        self, wither_rate=0.99, max_wither=0.01, gen=10000, top=100, score_func=None,
        rand=uniform_rand, b_rand=uniform_rand,
        input_count=1, input_act=lin,
        hidden_layers=3, hidden_neurons=10, hidden_act=ReLU,
        output_count=1, output_act=lin
    ):

    if score_func is not None:  # Scoring function must exist
      self.score_func = score_func
    else:
      raise ValueError('Model must have a scoring function.')

    self.gen = gen
    self.top = top
    self.mu = wither_rate
    self.max_wither = max_wither

    self.epochs = 0
    self.candidates = []
    self.net = None
    self.best_score = -float('inf')

    self.base_net = lambda: network(  # Create base network function using given parameters
        layers=None,
        rand=rand, b_rand=b_rand,
        input_count=input_count, input_act=input_act,
        hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, hidden_act=hidden_act,
        output_count=output_count, output_act=output_act
    )

    # Create RNG functions
    self.call_rand = lambda: create_rand(rand)
    self.call_rand_b = lambda: create_rand(b_rand)

  def fire(self, input):  # Generate output from the model
    if self.net is None:
      raise ValueError('Model has not been trained yet.')
    return self.net.fire(input)

  def train(self, gens=1, kill_old=False, cross_mode='pairwise', gamma_mode='complete', min_gamma=0.2, max_gamma=0.9, gamma_exp=0.0001, debug=True, visualizer=None):
    """
    MUTATION MODIFIERS

    gens = 1                 # Amount of generations to train
    kill_old = False         # Whether or not to keep last gen's top performers
    cross_mode = 'pairwise'  # 'random' means pair corresponding neurons,
                               'pairwise' means use distance to find a "better pair"
    gamma_mode = 'complete'  # 'none' means don't cross over at all,
                               'partial' means cross over but don't mutate if it happens,
                               'complete' means do everything (mutate + crossover)
    min_gamma = 0.2          # Minimum probability of crossover
    max_gamma = 0.9          # Maximum probability of crossover
    gamma_exp = 0.0001       # Exponential rate at which gamma increases
    """

    for g in range(gens):
      # Handle debug messages
      if debug:
        print(f'Training: [{g} / {gens}] generations. (Epoch {self.epochs})')

      if len(self.candidates) > 0:  # Case 1: A generation has already passed
        new_gen = []
        mult = max(pow(self.mu, self.epochs), self.max_wither)  # Scale based on previous epochs (dissipate)

        if gamma_mode == 'none':
          prob = 0
        else:
          psi = 1 - math.exp(self.epochs * -gamma_exp) # Scale breeding rate based on epochs
          prob = min_gamma * (1 - psi) + psi * max_gamma

        # Determine size of the next generations (new members only)
        amt = self.gen - self.top
        if kill_old:
            amt = self.gen

        for i in range(amt):  # Generate new nets slightly altering previous winners
          new_net = random.choice(self.candidates).copy()
          can_mutate = True

          if random.random() < prob:  # Randomly decide whether to breed or mutate
            if gamma_mode != 'complete':
              can_mutate = False

            other_net = random.choice(self.candidates)  # Select other network to breed the original with
            layers1 = new_net.layers
            layers2 = other_net.layers
            for j in range(len(layers1)):
              if not new_net.layers[j].input:
                l1 = layers1[j]
                l2 = layers2[j]

                if cross_mode == 'random':
                  for k in range(len(l1.neurons)):
                    n1 = l1.neurons[k]
                    n2 = l2.neurons[k]
                    for l in range(len(n1.w)):
                      alpha = random.random()
                      n1.w[l] = alpha * n1.w[l] + (1 - alpha) * n2.w[l]  # Randomly interpolate between both networks' weights and bias
                    alpha = random.random()
                    n1.b = alpha * n1.b + (1 - alpha) * n2.b

                elif cross_mode == 'pairwise':
                  for n1 in l1.neurons:
                    best = float('inf')
                    n2 = None
                    for n in l2.neurons:
                      dist = 0
                      for k in range(len(n1.w)):
                        dist += (n1.w[k] - n.w[k]) ** 2
                      dist += (n1.b - n.b) ** 2
                      dist = math.sqrt(dist)
                      if dist < best:
                        best = dist
                        n2 = n
                    for l in range(len(n1.w)):
                      alpha = random.random()
                      n1.w[l] = alpha * n1.w[l] + (1 - alpha) * n2.w[l]  # Randomly interpolate between both networks' weights and bias
                    alpha = random.random()
                    n1.b = alpha * n1.b + (1 - alpha) * n2.b

          if can_mutate:
            for layer in new_net.layers:
              if not layer.input:
                for n in layer.neurons:
                  for j in range(len(n.w)):
                    n.w[j] += self.call_rand() * mult  # Add random variations to weights and bias
                  n.b += self.call_rand_b() * mult
          new_gen.append(new_net)

        if kill_old:
          self.candidates = new_gen  # If the old generation is removed, set current generation to the new one
        else:
          self.candidates.extend(new_gen)  # Add mutated nets to current candidates

      else:  # Case 2: First generation
        for i in range(self.gen):  # Fill with random nets
          self.candidates.append(self.base_net())

      tuples = self.score_func(self.candidates)  # Generate scored tuples in the form (network, score), where a higher score is better

      # Select top candidates ('top' of them)
      tuples.sort(key=lambda t: t[1], reverse=True)
      new_candidates = []
      for i in range(self.top):
        new_candidates.append(tuples[i][0])

      # Modify based on new top candidates and pass onto the next epoch
      self.candidates = new_candidates
      if kill_old and self.best_score < tuples[0][1]:
        self.net = self.candidates[0]
        self.best_score = tuples[0][1]
      elif not kill_old:
        self.net = self.candidates[0]
      self.epochs += 1

      if visualizer is not None:
        visualizer(self, kill_old)

    if debug:
        print('Training complete.')