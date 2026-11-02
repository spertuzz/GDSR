# GDSR

Welcome! If you're wondering what GDSR stands for, it's *Geometry Dash Ship Regression*. Yes, that's what it really is, but what is that, exactly?

## Introducing GD

If you don't know what *Geometry Dash* is, it's a rhythm-platforming game made by RobTop Games, where the player moves forward at a constant speed and must avoid deadly obstacles. For the most part, the game features only one input: clicking (on either a mouse or a key). Depending on the "game mode" being played, that single input has a different effect on the player.

The one relevant to this project is the *Ship* game mode, where clicking causes the player to gain upward thrust, and releasing the key causes the ship to fall down.

## What this actually is

GDSR is a single-variable regression algorithm based on this game mode. It essentially does the following:

- Creates a simulated space (2D plane) where the reference data points are plotted,
- Places a "ship" a bit to the left of all the data points to give it a slight starting margin,
- Makes a neural network control the ship on every tick or "frame" of the simulation (using GD-like controls), deciding whether or not to click,
- And traces the final path to find the regression curve, where the final predicted y-value is obtained.

So, it's a silly and overly convoluted approach to achieve something other algorithms can do more efficiently. Neat!

## One caveat

This project was not implemented using `numpy`, `pandas`, or any of those pre-existing structures. Part of my goal with this project was not only to create my own regression algorithm, but also to learn more about neural networks and neuroevolution and to implement all of these concepts from scratch. Thus, running it requires very specific files from this repository and unique variables I'll highlight later.

### Later is now *(Quickstart Guide)*

So, how do we run this thing? It's quite simple; first, you need to import both `GDSR` and `mathstuff`, and then create a regressor using the following command:

```python
from mathstuff import *
from GDSR import *

GD_model = GD_ship_regressor(
    x_values,        # Real x values for training
    y_values,        # Real y values for training (in the same order as the x values)
    gen=100,         # Given that this is a genetic algorithm, this is the amount of networks in a generation
    top=10,          # Amount in a generation that are used for crossover
    resolution=300,  # Amount of ticks or "frames" in the simulation
    scroll=1,        # Multiplier for the scroll speed (horizontal speed) of the ship
    grav=1,          # Multiplier for the force of gravity
    thrust=1,        # Multiplier for the up force
    margin=0.4,      # Multiplier for the additional left margin
    y_margin=1       # Absolute amount of vertical margin
)
```

Then, to train the model, simply run:

```python
GD_model.train(
    gens=500,               # Amount of generations to train
    kill_old=False,         # 'False' means keep the winners in the generation, 'True' means don't let them move on to the next generation
    cross_mode='pairwise',  # Decides how the neural networks are crossed. 'random' means corresponding weights are paired and 'pairwise' uses Euclidean distance to find "better" pairs
    gamma_mode='complete',  # Decides how often they cross and in which order. 'none' means they don't, 'partial' means they either cross over or mutate, 'complete' means they do both
    min_gamma=0,            # Minimum initial chance of crossover occuring
    max_gamma=1,            # Final, "limit-to-infinity" chance of crossover
    gamma_exp=0.01,         # Growth rate for gamma
    debug=False,            # Debug mode
    visualize=False         # Build-in visualizer
)
```

Afterwards, all that's left to evaluate it at a certain x value is to run `GD_model.fire(x)`. `GD_model.fire(2)`, for instance, would return the y-coordinate of the point on the regression curve at $x=2$ (a float).
