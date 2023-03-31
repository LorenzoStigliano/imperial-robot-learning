# Import some installed python modules.
import numpy as np
import skimage.transform
import pyglet


# Define the Environment class
class Environment:

    # Initialisation function to create a new environment.
    def __init__(self, window_size):
        # window_size is the size in screen pixels that the environment will be draw in.
        self.window_size = window_size
        # map_size is he size of the discrete terrain map that the robot moves in.
        # The robot's state is continuous, but the map is discrete because each element has its own dynamics.
        self.map_size = 100
        # init_state is the robot's initial state.
        self.init_state = None
        # goal_state is the state that the robot is attempting to reach.
        self.goal_state = None
        # terrain_map is an array containing the impedance/height of the terrain
        self.terrain_map = None
        # background_sprite is an image of the map that is drawn on the screen.
        self.background_sprite = None
        # Generate the terrain map.
        self.generate_terrain_map()
        # Generate the background sprite.
        self.generate_background_sprite()
        # Generate the robot's initial state and goal state.
        self.generate_init_and_goal_states()

    # Function to generate the robot's initial and goal states in this environment.
    def generate_init_and_goal_states(self):
        # Set the initial state.
        self.init_state = np.array([0.1, 0.1])
        # Set the goal state
        self.goal_state = np.array([0.9, 0.9])

    # Function to generate the environment's terrain which the robot travels over.
    def generate_terrain_map(self):
        # Create a map of the environment.
        # Here, terrain_map[pos_across, pos_down] stores the impedance/height of the terrain, between 0 and 1.
        # A lower value in terrain_map means that the robot can travel faster over that position.
        self.terrain_map = np.zeros([self.map_size, self.map_size], dtype=np.float32)
        # Set the values for the map.
        for y in range(0, 100):
            impedance = (y + 1) / 100
            for x in range(0, 100):
                self.terrain_map[x, y] = impedance
        # Add an obstacle.
        for y in range(40, 50):
            for x in range(5, 70):
                self.terrain_map[x, y] = 1.0

    # Function to generate a sprite image of the environment, which pyglet will then display as the background image.
    # Do not edit this function.
    def generate_background_sprite(self):
        # Swap the axes so that the map can be used as an image.
        # This is because the map is [across, down], whereas the image is [down, across].
        terrain_image = np.swapaxes(self.terrain_map, 0, 1)
        # Rescale the map to the window size.
        terrain_image = skimage.transform.resize(terrain_image, (self.window_size, self.window_size), order=0)
        # Convert the greyscale image to an RGB image.
        terrain_image = (np.repeat(terrain_image[:, :, np.newaxis], 3, axis=2) * 255).astype(np.uint8)
        # Create a pyglet image from the numpy array.
        terrain_image = pyglet.image.ImageData(terrain_image.shape[1], terrain_image.shape[0], 'RGB', terrain_image.tobytes())
        # Create a sprite from this image, which will be displayed as the background during rendering.
        self.background_sprite = pyglet.sprite.Sprite(terrain_image)

    # Function to draw the environment.
    # Do not edit this function.
    def draw(self):
        # Draw the background sprite (i.e. the image showing the terrain).
        self.background_sprite.draw()
        # Draw the initial state as a blue square.
        init_width = init_height = 0.07 * self.window_size
        init_x = self.init_state[0] * self.window_size - 0.5 * init_width
        init_y = self.init_state[1] * self.window_size - 0.5 * init_width
        init_colour = (30, 30, 255)
        blue_square = pyglet.shapes.Rectangle(x=init_x, y=init_y, width=init_width, height=init_height, color=init_colour)
        blue_square.draw()
        # Draw the goal state as a green star.
        goal_x = self.goal_state[0] * self.window_size
        goal_y = self.goal_state[1] * self.window_size
        inner_radius = 0.025 * self.window_size
        outer_radius = 0.05 * self.window_size
        goal_colour = (0, 200, 0)
        green_star = pyglet.shapes.Star(x=goal_x, y=goal_y, outer_radius=outer_radius, inner_radius=inner_radius, num_spikes=5, color=goal_colour)
        green_star.draw()

    # The dynamics function, which returns the robot's next state given its current state and current action.
    def step(self, robot_current_state, robot_action):
        # Calculate the discrete coordinates in the map.
        map_x = int(robot_current_state[0] * self.map_size)
        map_y = int(robot_current_state[1] * self.map_size)
        # Get the impedance for these coordinates.
        # The impedance is between 0 and 1 and defines how quickly the robot can move at these coordinates.
        impedance = np.power(self.terrain_map[map_x, map_y], 0.2)
        # Calculate the next state.
        robot_next_state = robot_current_state + (1 - impedance) * robot_action
        # Clip the next state so that the robot stays within the environment.
        robot_next_state[0] = min(max(robot_next_state[0], 0), 0.99)
        robot_next_state[1] = min(max(robot_next_state[1], 0), 0.99)
        # Return the next state.
        return robot_next_state
