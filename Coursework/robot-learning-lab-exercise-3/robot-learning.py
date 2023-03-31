# Import some installed modules.
import time
import numpy as np
import pyglet

# Import some modules from this exercise.
from environment import Environment
from robot import Robot
import graphics


# Set the numpy random seed
seed = int(time.time())
np.random.seed(seed)

# Create a pyglet window.
window = pyglet.window.Window(width=graphics.window_size, height=graphics.window_size)
# Set the background colour to black.
pyglet.gl.glClearColor(0, 0, 0, 1)

# Create an environment.
environment = Environment(graphics.window_size)
# Create a robot.
robot = Robot(environment)


# Define what happens when the rendering is called.
@window.event
def on_draw():
    # Clear the window by filling with the background colour (black).
    window.clear()
    # Draw the environment.
    graphics.draw_environment(environment)
    # Draw whatever visualisations you would like to show on the window.
    graphics.draw_visualisations(robot)
    # Draw the robot.
    graphics.draw_robot(robot)
    # Optionally, save an image of the current window.
    # This may be helpful for the coursework.
    if 0:
        graphics.save_image()


# Define what happens on each timestep.
def update(dt):
    # Trigger the robot to calculate the next action.
    action = robot.next_action_random()
    # Execute this action in the environment.
    robot.state = environment.dynamics(robot.state, action)


# Set how frequently the update function is called.
pyglet.clock.schedule_interval(update, 1/100)


# Finally, call the main pyglet event loop.
# This will continually update and render the environment in a loop.
pyglet.app.run()
