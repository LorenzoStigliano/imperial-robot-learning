# Import some installed modules.
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
        self.init_state = np.array([0.1, 0.1], dtype=np.float32)
        # Set the goal state
        self.goal_state = np.array([0.9, 0.9], dtype=np.float32)

    # Function to generate the environment's terrain which the robot travels over.
    def generate_terrain_map(self):
        # Create a map of the environment.
        # Here, terrain_map[pos_across, pos_down] stores the impedance/height of the terrain, between 0 and 1.
        # A lower value in terrain_map means that the robot can travel faster over that position.
        self.terrain_map = np.zeros([self.map_size, self.map_size], dtype=np.float32)
        # Add two obstacles.
        for y in range(0, 30):
            for x in range(20, 50):
                self.terrain_map[x, y] = 0.3
        for y in range(70, 100):
            for x in range(50, 80):
                self.terrain_map[x, y] = 0.7


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

    # The dynamics of the environment.
    def dynamics(self, robot_current_state, robot_action):
        # Calculate the discrete coordinates in the map.
        map_x = int(robot_current_state[0] * self.map_size)
        map_y = int(robot_current_state[1] * self.map_size)
        # Get the impedance for these coordinates.
        # The impedance is between 0 and 1 and defines how quickly the robot can move at these coordinates.
        impedance = self.terrain_map[map_x, map_y]
        # Calculate the next state.
        robot_next_state = robot_current_state + (1 - impedance) * robot_action
        # Clip the next state so that the robot stays within the environment.
        robot_next_state[0] = min(max(robot_next_state[0], 0), 0.999)
        robot_next_state[1] = min(max(robot_next_state[1], 0), 0.999)
        # If the next state is within an obstacle, then cancel this action.
        map_next_x = int(robot_next_state[0] * self.map_size)
        map_next_y = int(robot_next_state[1] * self.map_size)
        next_impedance = self.terrain_map[map_next_x, map_next_y]
        if next_impedance == 1.0:
            robot_next_state = robot_current_state
        # Return the next state.
        return robot_next_state
