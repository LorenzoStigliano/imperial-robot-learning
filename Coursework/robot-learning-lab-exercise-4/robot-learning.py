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
window = pyglet.window.Window(width=graphics.window_size + 100, height=graphics.window_size)
# Set the background colour to black.
pyglet.gl.glClearColor(0, 0, 0, 1)

# Create an environment.
environment = Environment(graphics.window_size)
# Create a robot.
robot = Robot(environment)

# Set some flags which determine what is displayed on the window.
draw_terrain = True
draw_paths = True
draw_model = True

# Set the timer
start_time = time.time()

# Set the number of physical steps taken so far
num_steps = 0

# Set whether we are in learning or testing mode.
is_learning = True

# Define the time limit
TIME_LIMIT = 300


# Define what happens when the button is pressed
@window.event
def on_mouse_press(x, y, button, modifiers):
    global draw_terrain, draw_paths, draw_model
    # Check if the GUI buttons have been pressed
    if button == pyglet.window.mouse.LEFT:
        if graphics.button_x < x < graphics.button_x + graphics.button_width:
            if graphics.terrain_button_y < y < graphics.terrain_button_y + graphics.button_height:
                if draw_terrain:
                    draw_terrain = False
                else:
                    draw_terrain = True
            if graphics.paths_button_y < y < graphics.paths_button_y + graphics.button_height:
                if draw_paths:
                    draw_paths = False
                else:
                    draw_paths = True
            elif graphics.model_button_y < y < graphics.model_button_y + graphics.button_height:
                if draw_model:
                    draw_model = False
                else:
                    draw_model = True


# Define what happens when the rendering is called.
@window.event
def on_draw():
    # Clear the window by filling with the background colour (black).
    window.clear()
    # Draw the environment.
    graphics.draw_world(environment, robot, draw_terrain)
    # Draw the stored paths.
    if draw_paths:
        graphics.draw_paths(paths=robot.paths_to_draw)
    # Draw the model.
    if draw_model:
        graphics.draw_model(model=robot.model, action=np.array([environment.max_action, environment.max_action]))
    # Draw the buttons
    graphics.draw_buttons(draw_paths, draw_model, draw_terrain)
    # Optionally, save an image of the current window.
    # This may be helpful for the coursework.
    if 0:
        graphics.save_image()


# Define what happens on each timestep.
def step(dt):
    global is_learning, num_steps
    if is_learning:
        # Trigger the robot to calculate the next action.
        action = robot.next_action_learning()
        # Execute this action in the environment.
        next_state = environment.step(robot.state, action)
        # Send this information back to the robot.
        robot.process_transition(robot.state, action, next_state)
        # Compute the time remaining
        time_now = time.time()
        cpu_time = time_now - start_time
        action_time = num_steps
        time_elapsed = cpu_time + action_time
        time_remaining = TIME_LIMIT - time_elapsed
        print(f'Time remaining = {time_remaining}')
        num_steps += 1
        if time_remaining <= 0:
            print('Testing')
            is_learning = False
    else:
        # Trigger the robot to calculate the next action.
        action = robot.next_action_testing()
        # Execute this action in the environment.
        next_state = environment.step(robot.state, action)
        # Send this information back to the robot.
        robot.process_transition(robot.state, action, next_state)


# Set how frequently the update function is called.
pyglet.clock.schedule_interval(step, 0.001)


# Finally, call the main pyglet event loop.
# This will continually update and render the environment in a loop.
pyglet.app.run()
