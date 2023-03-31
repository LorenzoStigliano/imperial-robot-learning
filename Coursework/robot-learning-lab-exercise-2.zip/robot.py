# Import some installed modules.
import numpy as np
import scipy

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
        # The length of the trajectory calculated during planning.
        self.planning_horizon = 100
        # The actions calculated in the current plan.
        self.planned_actions = None
        # The timestep of the next action in the plan to execute.
        self.plan_timestep = 0
        # A list of paths that should be drawn to the screen.
        self.paths_to_draw = []

    # Function to convert a scalar angle, to an action parameterised by a 2-dimensional [x, y] direction.
    def convert_angle_to_action(self, angle):
        action_x = self.max_action * np.cos(angle)
        action_y = self.max_action * np.sin(angle)
        action = np.array([action_x, action_y])
        return action

    # Function to convert action parameterised by a 2-dimensional [x, y] direction to a scalar angle
    def convert_action_to_angle(self, action):
        return np.arctan2(action[1], action[0])

    # Function to compute the next action.
    def next_action(self):
        # Check if the robot needs to do planning.
        if self.needs_planning:
            # Do some planning.
            self.planned_actions = self.planning_cross_entropy(self.planning_horizon)
            # Set the flag so that the robot knows it has already done the planning.
            self.needs_planning = False
        # Check if the robot has any more actions left to execute in the plan.
        if self.plan_timestep < self.planning_horizon:
            # If there are more actions, return the next action.
            next_action = self.planned_actions[self.plan_timestep]
            # Increment the timestep in the plan.
            self.plan_timestep += 1
        # If there are no actions left in the plan, then return a 0 action.
        else:
            next_action = np.array([0.0, 0.0])
        # Return the next action
        return next_action


    # Function to compute the optimal action to take, given the robot's current state.
    def planning_random_shooting(self, planning_horizon):
        
        def reward_function(planned_states):
            goal_state = self.environment.goal_state
            return -np.linalg.norm(goal_state - planned_states[-1])
        
        best_planned_actions = []
        best_planned_states = []
        all_paths = []
        best_reward_value = -np.inf
        number_of_samples = 100

        for _ in range(number_of_samples):
            # Create an empty array to store the planned actions.
            planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
            # Create an empty array to store the planned states.
            planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
            # Set the initial state in the planning to the robot's current state.
            planning_state = self.state
            # Loop over the planning horizon.
            for i in range(planning_horizon):
                # Choose a random action.
                angle = np.random.uniform(0, 2 * 3.141592)
                action = self.convert_angle_to_action(angle)
                # Simulate the next state using the model.
                planning_state = self.model(planning_state, action)
                # Add this action to the array of planned actions.
                planned_actions[i] = action
                # Add this state to the array of planned states.
                planned_states[i] = planning_state
                
            #Check if new path improves the current best solution
            reward_value = reward_function(planned_states)
            if reward_value > best_reward_value:
                best_planned_actions = planned_actions
                best_planned_states = planned_states
                best_reward_value = reward_value
                all_paths.append(best_planned_states)
            else:
                all_paths.insert(0, planned_states)
        
        for states in all_paths[:-1]:
            # Create a path for these states, add it to the list of paths to be drawn.
            path = graphics.Path(states, (255, 120, 0), 2, 20)
            self.paths_to_draw.append(path)
        path = graphics.Path(best_planned_states, (255, 0, 0), 2, 20)
        self.paths_to_draw.append(path)
        # Return the array of actions
        return best_planned_actions

    def circmean(self, all_planned_angles, low=0, high=2 * np.pi):
        all_planned_angles = np.array(all_planned_angles)
        angles = np.mod(all_planned_angles - low, high - low) + low
        x = np.mean(np.cos(angles), axis=0)
        y = np.mean(np.sin(angles), axis=0)
        mean = np.arctan2(y, x)
        mean[mean<0] += 2 * np.pi
        return mean

    def circvar(self, angles, low=0, high=2*np.pi):
        angles = np.asarray(angles)
        angles = np.where(angles >= low, angles, angles + 2*np.pi)
        angles = np.where(angles < high, angles, angles - 2*np.pi)
        mean_angle = np.angle(np.mean(np.exp(1j*angles), axis=0))
        return 1 - np.abs(np.mean(np.exp(1j*(angles - mean_angle)), axis=0))

    # Function to compute the optimal action to take, given the robot's current state.
    def planning_cross_entropy(self, planning_horizon):

        def reward_function(planned_states):
            goal_state = self.environment.goal_state
            return -np.linalg.norm(goal_state - planned_states[-1])
        
        number_of_samples = 100
        number_of_refits = 5
        k = 0.03
        g = 200
        num_elements = int(k*number_of_samples)
        mean_angle = np.zeros([planning_horizon], dtype=np.float32)
        covariance_angle = np.diag(planning_horizon*[1000])
        
        for _ in range(number_of_refits):
            all_planned_angles = []
            all_planned_states = []
            all_reward_value = []
            for _ in range(number_of_samples):
                # Create an empty array to store the planned actions.
                planned_angles = np.random.multivariate_normal(mean_angle, covariance_angle)
                # Create an empty array to store the planned states.
                planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
                # Set the initial state in the planning to the robot's current state.
                planning_state = self.state
                # Loop over the planning horizon.
                for i in range(planning_horizon):
                    # Choose a random action.
                    angle = planned_angles[i]
                    action = self.convert_angle_to_action(angle)
                    # Simulate the next state using the model.
                    planning_state = self.model(planning_state, action)
                    # Add this state to the array of planned states.
                    planned_states[i] = planning_state                
                
                all_planned_angles.append(planned_angles)
                all_planned_states.append(planned_states)
        
            all_reward_value = [reward_function(planned_states) for planned_states in all_planned_states]    
            indices = np.argpartition(all_reward_value, -num_elements)[-num_elements:]
            best_planned_angles = np.array(all_planned_angles)[indices]

            # Create an empty array to store the planned states.
            planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
            mean_angle_all = self.circmean(all_planned_angles)
            
            # Set the initial state in the planning to the robot's current state.
            planning_state = self.state
            for i in range(planning_horizon):
                # Choose a random action.
                angle = mean_angle_all[i]
                action = self.convert_angle_to_action(angle)
                # Simulate the next state using the model.
                planning_state = self.model(planning_state, action)
                # Add this state to the array of planned states.
                planned_states[i] = planning_state

            # Add mean to found paths after each refits
            path = graphics.Path(planned_states, (255, g, 0), 1, 0)
            self.paths_to_draw.append(path)
            g -= 50
            print(self.circvar(best_planned_angles).shape)
            print(scipy.stats.circvar(best_planned_angles, axis=0).shape)
            print(np.mean(self.circvar(best_planned_angles) - scipy.stats.circvar(best_planned_angles, axis=0)))
            mean_angle = self.circmean(best_planned_angles)
            covariance_angle = np.diag(self.circvar(best_planned_angles))
        
        return planned_states

    # The dynamics model which the robot uses for planning.
    def model(self, robot_current_state, robot_action):
        # Here, we are saying that the model is the same as the true environment dynamics.
        next_state = self.environment.dynamics(robot_current_state, robot_action)
        return next_state

"""
all_planned_states = np.asarray(all_planned_states)[indices]
for states in all_planned_states:
    # Create a path for these states, add it to the list of paths to be drawn.
    path = graphics.Path(states, (255, 0, 0), 1, 20)
    self.paths_to_draw.append(path)

print(all_planned_actions.shape)

print(all_planned_actions.shape)
data = np.array([
    [[1,1],[2,2],[3,3]], 
    [[1,1],[2,2],[3,3]],
    [[1,1],[2,2],[3,3]],
    [[1,1],[2,2],[3,3]]
    ])
print(data.shape)
print(np.mean(data, axis=0))
print("var")
print(np.var(data, axis=0))
mean = np.mean(data, axis=0)
var = np.mean(data, axis=0)
print(data)
x = np.apply_along_axis(self.convert_action_to_angle, 1, mean)
var = np.apply_along_axis(self.convert_action_to_angle, 1, var)
print(x)
"""