# Import some installed python modules.
import time
import numpy as np
import pyglet

# Import some modules for this exercise.
from environment import Environment
from robot import Robot


# Define the window width and height (a square) in screen pixels.
# You may wish to modify this to fit your screen size.
WINDOW_SIZE = 500

# Set the numpy random seed
seed = int(time.time())
np.random.seed(seed)

# Create a pyglet window.
window = pyglet.window.Window(width=WINDOW_SIZE, height=WINDOW_SIZE)
# Set the background colour to black.
pyglet.gl.glClearColor(0, 0, 0, 1)

# Create an environment.
environment = Environment(WINDOW_SIZE)
# Create a robot.
robot = Robot(environment)


# Define what happens when the rendering is called.
@window.event
def on_draw():
    # Clear the window by filling with the background colour (black).
    window.clear()
    # Draw the environment.
    environment.draw()
    # Draw the robot.
    robot.draw()


# Define what happens when the state is updated.
def update(dt):
    # Trigger the robot to choose an action and then execute this action in the environment.
    robot.step(environment)


# Set how frequently the update function is called.
pyglet.clock.schedule_interval(update, 1/100)


# Finally, call the main pyglet event loop.
# This will continually update and render the environment in a loop.
pyglet.app.run()
