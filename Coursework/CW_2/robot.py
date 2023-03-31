# Import some installed modules.
import numpy as np

# Define the Robot class
class Robot:
    # Initialisation function to create a new robot
    def __init__(self, model, max_action, goal_state):
        
        # STATE AND ACTION DATA
        # The maximum magnitude of the robot's action. Do not edit this.
        self.max_action = max_action
        # The goal state
        self.goal_state = goal_state
        
        # MODEL DATA
        self.model = model
        
        # TRAINING DATA 
        # Keep track of previous state
        self.previous_state = np.array([np.inf, np.inf])
        # Set the episode length
        self.episode_length = 200
        # Set the number of steps taken so far in this episode.
        self.episode_num_steps = 0      
        # The actions calculated in the current plan.
        self.planned_actions = []
        # The planning horizon (number of actions in the plan)
        self.training_planning_horizon = 30
        # The number of sampled action sequences during planning
        self.training_num_samples = 400
        # Ensure we keep the real goal state for testing
        self.goal_state_real = goal_state
        # Keep track of start state
        self.start_state = np.zeros([1, 2], dtype=np.float32)
        # Check if the robot needs to return to the start 
        self.returing = True
        
        # TESTING DATA 
        # The planning horizon (number of actions in the plan)
        self.testing_planning_horizon = 20
        # The number of sampled action sequences during planning
        self.testing_num_samples = 400
        # If testing is starting 
        self.testing = True

        # VISUALISATION DATA
        # A list of paths that should be drawn to the screen.
        self.paths_to_draw = []

    # Function to compute the next action, during the training phase.
    def next_action_training(self, state):
        
        # Ensure we get the start state of the robot
        if not self.start_state.any():
            self.start_state = state
        
        # Only reset if the robot get stuck (impedence 1)
        if all(self.previous_state == state):
            self.returing = True
            self.goal_state = self.goal_state_real 
            self.previous_state = np.array([10,10])
            return np.zeros(2), True
        else:
            self.previous_state = state
        
        # Add one to episode number of step counter
        self.episode_num_steps +=1 
        
        # Check if at end of episode or reached goal
        if self.episode_num_steps == self.episode_length or np.linalg.norm(state - self.goal_state) < 0.03:
            
            if self.returing:
                self.goal_state = self.start_state
            else: 
                self.goal_state = self.goal_state_real 
            
            self.returing = not self.returing
            self.episode_num_steps = 0

            return self.random_action(), False
        
        # Check if within 0.15 of goal
        if np.linalg.norm(state - self.goal_state) < 0.15:
            self.planned_actions = self.planning_cross_entropy(
                state = state, 
                planning_horizon = 10, 
                number_of_samples = 100, 
                number_of_refits = 2, 
                k = 0.1)

        # Check if planned actions length = 0 we need to replan
        if len(self.planned_actions) == 0:
            self.planned_actions = self.planning_cross_entropy(
                state = state, 
                planning_horizon = self.training_planning_horizon, 
                number_of_samples = self.training_num_samples, 
                number_of_refits = 2, 
                k = 0.3)

        next_action = self.planned_actions.pop(0)
        
        return np.array(next_action), False

    # Function to compute the next action, during the testing phase.
    def next_action_testing(self, state):
        
        # Ensure the goal_state is the real goal state for testing 
        self.goal_state = self.goal_state_real
        
        # Check if we start to test
        if self.testing:            
            self.planned_actions = self.planning_cross_entropy(
                state = state, 
                planning_horizon = self.testing_planning_horizon, 
                number_of_samples = self.testing_num_samples, 
                number_of_refits = 3, 
                k = 0.2)
            self.testing = False
            self.episode_num_steps = 0

        self.episode_num_steps +=1 
  
        # Check if within 0.15 of goal
        if np.linalg.norm(state - self.goal_state) < 0.15:
            self.planned_actions = self.planning_cross_entropy(
                state = state, 
                planning_horizon = 10, 
                number_of_samples = 100, 
                number_of_refits = 2, 
                k = 0.1)
        
        # Check if planned actions length = 0 we need to replan
        if len(self.planned_actions) == 0:
            self.planned_actions = self.planning_cross_entropy(
                state = state, 
                planning_horizon = self.testing_planning_horizon, 
                number_of_samples = self.testing_num_samples, 
                number_of_refits = 3, 
                k = 0.2)
        
        next_action = self.planned_actions.pop(0)
        
        return np.array(next_action)

    # Function to compute the next action.
    def random_action(self):
        # Choose a random action.
        action_x = np.random.uniform(-self.max_action, self.max_action)
        action_y = np.random.uniform(-self.max_action, self.max_action)
        action = np.array([action_x, action_y])
        # Return this random action.
        return action

    def process_transition(self, state, action):
        self.model.update_uncertainty(state, action)

    # Function to calulate the reward 
    def reward_function(self, state, planned_states):
        # Dense reward
        if np.linalg.norm(self.goal_state - state) < 0.2:
            return -np.linalg.norm(self.goal_state - planned_states)
        # Sparse reward
        else:
            return -np.linalg.norm(self.goal_state - planned_states[-1]) 

    # Function to convert a scalar angle, to an action parameterised by a 2-dimensional [x, y] direction.
    def convert_angle_to_action(self, angle):
        action_x = self.max_action * np.cos(angle)
        action_y = self.max_action * np.sin(angle)
        action = np.array([action_x, action_y])
        return action
    
    # Function to calulate the circular mean, like the scipy implementation 
    def circmean(self, all_planned_angles, low=0, high=2*np.pi):
        # This is done to handle circular mean values, which can be outside the range of 0 to 2pi
        angles = np.mod(all_planned_angles - low, high - low) + low
        # Calculate the mean cosine and sine of the shifted angles 
        x = np.mean(np.cos(angles), axis=0)
        y = np.mean(np.sin(angles), axis=0)
        # Calcualte angle in radians
        mean = np.arctan2(y, x)
        # Add 2pi to the angle if necessary so all in the desired range
        mean[mean<0] += 2 * np.pi
        return mean

    # Function to calulate the circular variance, like the scipy implementation   
    def circvar(self, all_planned_angles, low=0, high=2*np.pi):
        # This is done to handle circular mean values, which can be outside the range of 0 to 2pi
        angles = np.mod(all_planned_angles - low, high - low) + low
        # Converts the complex average of the angles into an angle between -pi and pi.
        mean_angle = np.angle(np.mean(np.exp(1j*angles), axis=0))
        # Calculate the circular variance using the formula 1 - R, 
        # Where R is the magnitude of the complex average of the angles shifted by the mean angle
        return 1 - np.abs(np.mean(np.exp(1j*(angles - mean_angle)), axis=0))

    # Function to compute the optimal action to take, given the robot's current state.
    def planning_cross_entropy(self, state, planning_horizon, number_of_samples, number_of_refits, k):
        
        num_elements = int(k*number_of_samples)
        mean_angle = np.zeros([planning_horizon], dtype=np.float32)
        covariance_angle = np.diag(planning_horizon*[200])

        for _ in range(number_of_refits):
            
            all_planned_angles = []
            all_planned_states = []

            for _ in range(number_of_samples):
                # Create an empty array to store the planned actions.
                planned_angles = np.random.multivariate_normal(mean_angle, covariance_angle)
                # Create an empty array to store the planned states.
                planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
                # Set the initial state in the planning to the robot's current state.
                planning_state = state
                # Loop over the planning horizon.
                for i in range(planning_horizon):
                    # Choose a random action.
                    angle = planned_angles[i]
                    action = self.convert_angle_to_action(angle)
                    # Simulate the next state using the model.
                    planning_state, _ = self.model.predict(planning_state, action)
                    # Add this state to the array of planned states.
                    planned_states[i] = planning_state          
                
                all_planned_angles.append(planned_angles)
                all_planned_states.append(planned_states)
        
            # Calculate reward function for the planned states
            all_reward_value = [self.reward_function(state, planned_states) for planned_states in all_planned_states]    
            # Get indicies of the best rewards in descending order
            indices = np.argpartition(all_reward_value, -num_elements)[-num_elements:]
            # Get the best planned angles
            best_planned_angles = np.array(all_planned_angles)[indices]
            # Update mean and covariance
            mean_angle = self.circmean(best_planned_angles)
            covariance_angle = np.diag(self.circvar(best_planned_angles))
            
        # Get final planned actions
        planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
        for i in range(planning_horizon):
            angle = mean_angle[i]
            action = self.convert_angle_to_action(angle)
            planned_actions[i] = action
        
        return list(planned_actions)