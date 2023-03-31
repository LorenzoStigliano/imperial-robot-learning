# Import some installed modules.
import numpy as np

# Import the model module.
import model


# Define the Robot class
class Robot:

    # Initialisation function to create a new robot
    def __init__(self, environment):
        # STATE AND ACTION DATA
        # The maximum magnitude of the robot's action. Do not edit this.
        self.max_action = environment.max_action
        # The initial state
        self.init_state = environment.init_state
        # The goal state
        self.goal_state = environment.goal_state
        # The robot's current state.
        self.state = self.init_state
        # MODEL DATA
        self.model = model.Model(environment)
        # PLANNING DATA
        # VISUALISATION DATA
        # A list of paths that should be drawn to the screen.
        self.paths_to_draw = []

    # Function to compute the next action, during the learning phase.
    def next_action_learning(self):
        # Choose a random action.
        action_x = np.random.uniform(-self.max_action, self.max_action)
        action_y = np.random.uniform(-self.max_action, self.max_action)
        action = np.array([action_x, action_y])
        # Return the next action
        return action

    # Function to compute the next action, during the testing phase.
    def next_action_testing(self):
        # Choose a random action.
        action_x = np.random.uniform(-self.max_action, self.max_action)
        action_y = np.random.uniform(-self.max_action, self.max_action)
        action = np.array([action_x, action_y])
        # Return the next action
        return action

    # Function to process transition data.
    # Do not edit this function.
    def process_transition(self, state, action, next_state):
        self.model.update_uncertainty(state, action)
        self.state = next_state
