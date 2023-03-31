# Import some installed modules.
import pyglet
import os


# Define the window width and height (a square) in screen pixels.
# You may wish to modify this to fit your screen size.
window_size = 500


# Class to store a path which will then be drawn to the window.
class Path:

    # Initialisation function.
    def __init__(self, points, colour, width, skip=0):
        self.points = points
        self.colour = colour
        self.width = width
        self.skip = skip


# Function to draw the robot on the window.
def draw_robot(robot):
    robot_x = robot.state[0] * window_size
    robot_y = robot.state[1] * window_size
    robot_radius = 0.025 * window_size
    robot_colour = (200, 0, 0)
    red_circle = pyglet.shapes.Circle(x=robot_x, y=robot_y, radius=robot_radius, color=robot_colour)
    red_circle.draw()


# Function to draw the environment on the window.
def draw_environment(environment):
    # Draw the background sprite (i.e. the image showing the terrain).
    environment.background_sprite.draw()
    # Draw the initial state as a blue square.
    init_width = init_height = 0.07 * window_size
    init_x = environment.init_state[0] * window_size - 0.5 * init_width
    init_y = environment.init_state[1] * window_size - 0.5 * init_width
    init_colour = (30, 30, 255)
    blue_square = pyglet.shapes.Rectangle(x=init_x, y=init_y, width=init_width, height=init_height, color=init_colour)
    blue_square.draw()
    # Draw the goal state as a green star.
    goal_x = environment.goal_state[0] * window_size
    goal_y = environment.goal_state[1] * window_size
    inner_radius = 0.025 * window_size
    outer_radius = 0.05 * window_size
    goal_colour = (0, 200, 0)
    green_star = pyglet.shapes.Star(x=goal_x, y=goal_y, outer_radius=outer_radius, inner_radius=inner_radius, num_spikes=5, color=goal_colour)
    green_star.draw()


# Function to draw other visualisations on the window, e.g. the robot's plan, or anything else you want.
def draw_visualisations(robot):
    # Draw whatever paths you want to drawn from the robot's planning.
    draw_paths(robot.paths_to_draw, skip=0)


# Function to draw some paths on the window.
# The skip argument allows you to speed up drawing by only drawing some of the points.
def draw_paths(paths, skip=0):
    batch = pyglet.graphics.Batch()
    batch_lines = []
    for path in paths:
        for i in range(len(path.points) - 1 - skip):
            point_1 = window_size * path.points[i]
            point_2 = window_size * path.points[i + 1 + skip]
            line_segment = pyglet.shapes.Line(point_1[0], point_1[1], point_2[0], point_2[1], width=path.width, color=path.colour, batch=batch)
            batch_lines.append(line_segment)
            i += skip
    batch.draw()


# Save an image of the current window.
def save_image():
    # If the program is quit during the saving process, the image file will be corrupted.
    # Therefore, we save it with a temporary filename.
    # Then when we are sure that saving has completed, we rename it to the desired filename.
    # With this, if the program quits during the saving process, the previous image with this filename will persist.
    filename = 'robot-learning.png'
    path = os.path.join(os.getcwd(), filename)
    path_temp = os.path.join(os.getcwd(), 'temp.png')
    pyglet.image.get_buffer_manager().get_color_buffer().save(path_temp)
    os.rename(path_temp, path)
