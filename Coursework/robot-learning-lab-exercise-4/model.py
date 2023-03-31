import numpy as np
from scipy.spatial import KDTree


class Model:

    def __init__(self, environment):
        self.environment = environment
        self.kd_tree = None
        self.transitions = np.empty(shape=[4])

    # Make a prediction of the next state
    def predict(self, state, action):
        if self.kd_tree is None:
            return state + np.random.uniform(-0.01, 0.01, 2)
        # True dynamics
        true_next_state = self.environment.dynamics(state, action)
        # Get the uncertainty
        query = np.array([state[0], state[1], action[0], action[1]])
        distance, _ = self.kd_tree.query(query)
        uncertainty = 0.02 * np.sqrt(distance)
        # Compute the predicted next state with uncertainty
        predicted_next_state = true_next_state
        predicted_next_state[0] += np.random.normal(loc=0, scale=uncertainty)
        predicted_next_state[1] += np.random.normal(loc=0, scale=uncertainty)
        # Return this predicted state
        return predicted_next_state

    # Add a transition and update the uncertainty
    def update_uncertainty(self, state, action):
        new_transition = np.array([state[0], state[1], action[0], action[1]])
        self.transitions = np.vstack((self.transitions, new_transition))
        self.kd_tree = KDTree(self.transitions)
