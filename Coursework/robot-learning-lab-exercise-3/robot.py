# Import some installed modules.
import numpy as np

# Import some modules from this exercise.
import graphics


# Define the Robot class
class Robot:

    # Initialisation function to create a new robot
    def __init__(self, environment):
        # Set the environment.
        self.environment = environment
        # Set the window size, which is necessary when the robot is drawn to the screen.
        self.window_size = environment.window_size
        # The maximum magnitude of the robot's action. Do not edit this.
        self.max_action = 0.05
        # The robot's initial state.
        self.init_state = environment.init_state
        # The robot's current_state.
        self.state = self.init_state
        # A flag to set whether or not the robot needs to do planning.
        self.needs_planning = True
        # The actions calculated in the current plan.
        self.planned_actions = None
        # The planning horizon (number of actions in the plan)
        self.planning_horizon = 50
        # The number of sampled action sequences during planning
        self.num_samples = 100
        # The timestep of the next action in the plan to execute.
        self.plan_timestep = 0
        # Set the episode length
        self.episode_length = 100
        # Set the number of steps taken so far in this episode.
        self.episode_num_steps = 0
        # A list of paths that should be drawn to the screen.
        self.paths_to_draw = []

    # Function to convert a scalar angle, to an action parameterised by a 2-dimensional [x, y] direction.
    def convert_angle_to_action(self, angle):
        action_x = self.max_action * np.cos(angle)
        action_y = self.max_action * np.sin(angle)
        action = np.array([action_x, action_y])
        return action

    # Function to compute the next action.
    def next_action_open_loop(self):
        # Check if the episode has finished.
        if self.episode_num_steps == self.episode_length:
            self.state = self.init_state
            self.episode_num_steps = 0
        # Check if the robot needs to do planning.
        if self.needs_planning:
            # Do some planning.
            self.planned_actions = self.planning_random_shooting()
            # Set the flag so that the robot knows it has already done the planning.
            self.needs_planning = False
        # Check if the robot has any more actions left to execute in the plan.
        if self.plan_timestep < len(self.planned_actions):
            # If there are more actions, return the next action.
            next_action = self.planned_actions[self.plan_timestep]
            # Increment the timestep in the plan.
            self.plan_timestep += 1
        # If there are no actions left in the plan, then return a 0 action.
        else:
            next_action = np.array([0.0, 0.0])
        # Increment the number of steps in this episode.
        self.episode_num_steps += 1
        # Return the next action
        return next_action

    # Function to compute the next action.
    def next_action_random(self):
        # Check if the episode has finished.
        if self.episode_num_steps == self.episode_length:
            self.state = self.init_state
            self.episode_num_steps = 0
        # Choose a random action.
        angle = np.random.uniform(0, 2 * 3.141592)
        action = self.convert_angle_to_action(angle)
        # Increment the number of steps in this episode.
        self.episode_num_steps += 1
        # Return this random action.
        return action

    # Function to perform random shooting planning.
    def planning_random_shooting(self):
        # Create an empty array to store the sampled actions.
        sampled_actions = np.zeros([self.num_samples, self.planning_horizon + 1, 2], dtype=np.float32)
        # Create an empty array to store the sampled states.
        sampled_paths = np.zeros([self.num_samples, self.planning_horizon + 1, 2], dtype=np.float32)
        sampled_paths[:, 0] = self.state
        # Create a list of scores, one for each sample
        sample_scores = np.zeros(self.num_samples, dtype=np.float32)
        for sample_num in range(self.num_samples):
            # Set the initial state in the planning to the robot's current state.
            planning_state = self.state
            # Loop over the planning horizon.
            for planning_step in range(self.planning_horizon):
                # Choose a random action.
                angle = np.random.uniform(0, 2 * 3.141592)
                action = self.convert_angle_to_action(angle)
                # Simulate the next state using the model.
                planning_state = self.model(planning_state, action)
                # Add this action to the array of planned actions.
                sampled_actions[sample_num, planning_step] = action
                # Add this state to the array of planned states.
                sampled_paths[sample_num, planning_step + 1] = planning_state
            # Calculate and store the score for this sample
            score = self.reward_function(sampled_paths[sample_num])
            sample_scores[sample_num] = score
        # Calculate the best sample
        best_sample = np.argmax(sample_scores)
        # Create the path to be drawn, showing the best sample
        path = graphics.Path(sampled_paths[best_sample], (255, 200, 0), 2, 0)
        self.paths_to_draw.append(path)
        # Return the best actions
        best_actions = sampled_actions[best_sample]
        return best_actions

    # The dynamics model which the robot uses for planning.
    def model(self, robot_current_state, robot_action):
        # Here, we are saying that the model is the same as the true environment dynamics.
        next_state = self.environment.dynamics(robot_current_state, robot_action)
        return next_state

    # The reward function used to score each sampled action sequence.
    def reward_function(self, path):
        reward = -np.linalg.norm(path[-1] - self.environment.goal_state)
        return reward
