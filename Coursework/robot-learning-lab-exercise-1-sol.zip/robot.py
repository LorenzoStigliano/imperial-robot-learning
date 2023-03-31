# Import some installed python modules.
import numpy as np
import pyglet
import pyglet.graphics
from pyglet.image import get_buffer_manager


# Define the Robot class
class Robot:

    # Initialisation function to create a new robot
    def __init__(self, environment):
        # Set the window size, which is necessary when the robot is drawn to the screen.
        self.window_size = environment.window_size
        # The maximum magnitude of the robot's action.
        # Do not edit this.
        self.max_action = 0.01
        # The robot's initial state.
        self.init_state = environment.init_state
        # The robot's current_state.
        self.state = self.init_state
        # The number of steps taken by the robot in the current episode.
        self.num_steps = 0
        # The current path.
        self.current_path = []
        # The best path so far.
        self.best_path = []
        # The best distance so far.
        self.best_distance = np.inf

    # Function to trigger the robot to take an action and then update its state.
    def step(self, environment):
        # Add this state to the current path.
        self.current_path.append(self.state)
        # Choose a random action
        angle = np.random.uniform(0, 2 * 3.141592)
        action_x = self.max_action * np.cos(angle)
        action_y = self.max_action * np.sin(angle)
        action = np.array([action_x, action_y])
        # Execute the action and update the robot's state.
        self.state = environment.step(self.state, action)
        # Increment the number of steps taken by the robot in this episode.
        self.num_steps += 1
        # Check if the episode should be reset.
        if self.num_steps == 100:
            # Add this state to the current path.
            self.current_path.append(self.state)
            # Calculate the distance between the robot and the goal.
            distance = np.linalg.norm(self.state - environment.goal_state)
            # Check if this is the best distance so far.
            if distance < self.best_distance:
                self.best_distance = distance
                # Save this path.
                self.best_path = self.current_path
            # Reset the episode
            self.state = self.init_state
            self.num_steps = 0
            self.current_path = []
            # Save the current image
            buffer = get_buffer_manager().get_color_buffer()
            buffer.save('best-path.png')

    # Function to draw the robot.
    def draw(self):
        # Draw the robot
        robot_x = self.state[0] * self.window_size
        robot_y = self.state[1] * self.window_size
        robot_radius = 0.025 * self.window_size
        robot_colour = (200, 0, 0)
        red_circle = pyglet.shapes.Circle(x=robot_x, y=robot_y, radius=robot_radius, color=robot_colour)
        red_circle.draw()
        # Draw the best path.
        # To speed up drawing, we only plot every 10 steps in the path.
        batch = pyglet.graphics.Batch()
        batch_lines = []
        for i in range(len(self.best_path) - 10):
            curr_step = self.window_size * self.best_path[i]
            next_step = self.window_size * self.best_path[i + 10]
            line = pyglet.shapes.Line(curr_step[0], curr_step[1], next_step[0], next_step[1], width=3, color=(255, 200, 0, 255), batch=batch)
            batch_lines.append(line)
            i += 10
        batch.draw()
