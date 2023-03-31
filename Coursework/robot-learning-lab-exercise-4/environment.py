# Import some installed modules.
import numpy as np
import skimage.transform
import pyglet
import time
from perlin_noise import PerlinNoise


# Define the Environment class
class Environment:

    # Initialisation function to create a new environment.
    def __init__(self, window_size):
        # window_size is the size in screen pixels that the environment will be draw in.
        self.window_size = window_size
        # map_size is he size of the discrete terrain map that the robot moves in.
        # The robot's state is continuous, but the map is discrete because each element has its own dynamics.
        self.map_size = 100
        # STATE DATA
        self.init_state = None
        # goal_state is the state that the robot is attempting to reach.
        self.goal_state = None
        # ENVIRONMENT DATA
        # terrain_map is an array containing the impedance/height of the terrain
        self.terrain_map = None
        # background_sprite is an image of the map that is drawn on the screen.
        self.background_sprite = None
        # ACTION DATA
        # The maximum action that a robot can execute in its x and y directions.
        self.max_action = 0.02
        # INITIALISATION FUNCTIONS
        # Generate the terrain map.
        self.generate_terrain_map()
        # Generate the background sprite.
        self.generate_background_sprite()
        # Generate the robot's initial state and goal state.
        self.generate_init_and_goal_states()

    # Function to generate the robot's initial and goal states in this environment.
    def generate_init_and_goal_states(self):
        # Define the initial state and goal state
        self.init_state = np.random.uniform(0.05, 0.95, 2)
        self.goal_state = np.random.uniform(0.05, 0.95, 2)
        # Ensure that the initial and goal states are sufficiently far apart
        while np.linalg.norm(self.init_state - self.goal_state) < 0.9:
            self.init_state = np.random.uniform(0.05, 0.95, 2)
            self.goal_state = np.random.uniform(0.05, 0.95, 2)

    # Function to generate the environment's terrain which the robot travels over.
    def generate_terrain_map(self):
        # Create a random map with Perlin noise
        num_octaves = 2
        noise = PerlinNoise(octaves=num_octaves, seed=int(time.time()))
        # Create a map of the environment
        terrain_map = np.zeros([100, 100], dtype=np.float32)
        # Populate this map with the noise
        for i in range(100):
            i_norm = i / 100
            for j in range(100):
                j_norm = j / 100
                terrain_map[i, j] = noise([j_norm, i_norm])
        # Create another random map with Perlin noise, at a different frequency
        num_octaves = 4
        noise = PerlinNoise(octaves=num_octaves, seed=int(time.time()))
        # Populate this map with the noise
        for i in range(100):
            i_norm = i / 100
            for j in range(100):
                j_norm = j / 100
                terrain_map[i, j] += noise([j_norm, i_norm])
        # Create another random map with Perlin noise, at a different frequency
        num_octaves = 6
        noise = PerlinNoise(octaves=num_octaves, seed=int(time.time()))
        # Populate this map with the noise
        for i in range(100):
            i_norm = i / 100
            for j in range(100):
                j_norm = j / 100
                terrain_map[i, j] += noise([j_norm, i_norm])
        # Add a random block
        x = int(np.random.uniform(20, 49))
        y = int(np.random.uniform(20, 49))
        width = int(np.random.uniform(30, 50))
        height = int(np.random.uniform(30, 50))
        value = 0.9 * np.max(terrain_map)
        for i in range(x, x + width):
            i_norm = i / 100
            for j in range(y, y + height):
                j_norm = j / 100
                terrain_map[i, j] = value
        # Normalise the map
        self.terrain_map = (terrain_map - np.min(terrain_map)) / (np.max(terrain_map) - np.min(terrain_map))

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

    # Take one timestep in the environment
    def step(self, robot_state, robot_action):
        robot_next_state = self.dynamics(robot_state, robot_action)
        return robot_next_state

    # Define the environment dynamics
    def dynamics(self, robot_state, robot_action):
        # Clip the action
        robot_action[0] = np.clip(robot_action[0], -self.max_action, self.max_action)
        robot_action[1] = np.clip(robot_action[1], -self.max_action, self.max_action)
        # Calculate the discrete coordinates in the map.
        map_x = int(robot_state[0] * self.map_size)
        map_y = int(robot_state[1] * self.map_size)
        map_x = np.clip(map_x, 0, 99)
        map_y = np.clip(map_y, 0, 99)
        # Get the impedance for these coordinates.
        # The impedance is between 0 and 1 and defines how quickly the robot can move at these coordinates.
        pow = 10
        impedance = np.power(self.terrain_map[map_x, map_y], pow)
        # Calculate the next state.
        speed = 1 - impedance
        robot_next_state = robot_state + speed * robot_action
        # Clip the next state so that the robot stays within the environment.
        robot_next_state[0] = np.clip(robot_next_state[0], 0, 0.99)
        robot_next_state[1] = np.clip(robot_next_state[1], 0, 0.99)
        return robot_next_state
