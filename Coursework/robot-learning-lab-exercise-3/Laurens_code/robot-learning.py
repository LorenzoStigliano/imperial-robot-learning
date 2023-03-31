# Import some installed modules.
import time
import numpy as np
import pyglet


# Import some modules from this exercise.
from environment import Environment
from robot import Robot
import graphics

from torch_example import Network

import torch
from matplotlib import pyplot as plt
# Turn on interactive mode for PyPlot, to prevent the displayed graph from blocking the program flow.
plt.ion()


# Set the numpy random seed
seed = int(time.time())
np.random.seed(seed)

# Create a pyglet window.
window = pyglet.window.Window(width=graphics.window_size, height=graphics.window_size)
# Set the background colour to black.
pyglet.gl.glClearColor(0, 0, 0, 1)

# Create an environment.
environment = Environment(graphics.window_size)
# Create a robot.
robot = Robot(environment)


# update function to train model
# Define what happens when the rendering is called.
@window.event
def on_draw():
    # Clear the window by filling with the background colour (black).
    window.clear()
    # Draw the environment.
    graphics.draw_environment(environment)
    # Draw whatever visualisations you would like to show on the window.
    graphics.draw_visualisations(robot)
    # Draw the robot.
    graphics.draw_robot(robot)
    # Optionally, save an image of the current window.
    # This may be helpful for the coursework.
    graphics.save_image()


def update_training_model(dt):

    buffer_data = np.zeros(shape = (5000, 4))
    buffer_labels = np.zeros(shape = (5000, 2))
    
    testing_data = np.zeros(shape = (5000, 4))
    testing_labels = np.zeros(shape = (5000, 2))
    
    for i in range(5000):
        action = robot.next_action_random()
        current_state = robot.state     
        robot.state = environment.dynamics(robot.state, action)
        next_state = robot.state
        buffer_data[i] = current_state[0], current_state[1], action[0], action[1]
        buffer_labels[i] = next_state[0], next_state[1]
        
    for j in range(5000):
        action = robot.next_action_random()
        current_state = robot.state     
        robot.state = environment.dynamics(robot.state, action)
        next_state = robot.state
        testing_data[j] = current_state[0], current_state[1], action[0], action[1]
        testing_labels[j] = next_state[0], next_state[1]
    
    #np.savetxt('buffer_data.txt', buffer_data)
    #np.savetxt('buffer_labels.txt', buffer_labels)
    
    #np.savetxt('testing_data.txt', testing_data)
    #np.savetxt('testing_labels.txt', testing_labels)
  
    
    dataset_inputs = torch.tensor(buffer_data.astype(np.float32))
    dataset_labels = torch.tensor(buffer_labels.astype(np.float32))
    testing_inputs = torch.tensor(testing_data.astype(np.float32))
    testing_labels = torch.tensor(testing_labels.astype(np.float32))
    
    N = 5000
    training_size = int(N * 0.9)
    
    # Initialise the network, loss function, and optimiser.
    network = Network()
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(network.parameters(), lr=0.01)

    # Divide the data into training and validation subsets.
    training_inputs = dataset_inputs[:training_size]
    training_labels = dataset_labels[:training_size]
    validation_inputs = dataset_inputs[training_size:]
    validation_labels = dataset_labels[training_size:]
    

    # Convert the data into tensors and create a DataLoader for each set.
    training_dataset = torch.utils.data.TensorDataset(training_inputs, training_labels)
    validation_dataset = torch.utils.data.TensorDataset(validation_inputs, validation_labels)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=10, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=False)
    
    testing_dataset = torch.utils.data.TensorDataset(testing_inputs, testing_labels)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=10, shuffle=True)
    
    # Training loop.
    training_losses = []
    validation_losses = []
    for epoch in range(50):
        # Training phase.
        network.train()
        training_loss = 0
        for inputs, labels in training_loader:
            optimiser.zero_grad()
            predictions = network(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimiser.step()
            training_loss += loss.item()
        training_loss /= len(training_loader)
        training_losses.append(training_loss)
    
        # Validation phase.
        network.eval()
        validation_loss = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                predictions = network(inputs)
                loss = criterion(predictions, labels)
                validation_loss += loss.item()
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)
    
        # Print the training and validation losses.
        print(f"[Epoch {epoch + 1}]\tTraining Loss: {training_loss:.4f}\tValidation Loss: {validation_loss:.4f}")
    
        # Plot and save the training and validation losses.
        plt.clf()
        plt.plot(range(epoch + 1), training_losses, label="Training loss", color=(0.8, 0.2, 0.2))
        plt.plot(range(epoch + 1), validation_losses, label="Validation loss", color=(0.2, 0.6, 0.2))
        plt.title("Loss Curve for Dummy Data")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.savefig("loss-curve.png")
        plt.pause(0.01)
        
    testing_loss = 0
    with torch.no_grad():
        for inputs, labels in testing_loader:
            predictions = network(inputs)
            loss = criterion(predictions, labels)
            testing_loss += loss.item()
        testing_loss /= len(testing_loader)
    
    print('Testing Loss', testing_loss)
    
    torch.save(network.state_dict(), 'my_model_3.pth')
    pyglet.app.exit()
    
    
open_loop_list_of_states = []
open_loop_list_of_actions = []

closed_loop_list_of_states = []
closed_loop_list_of_actions = []

def run_open_loop_planning():

        if len(open_loop_list_of_states) < robot.episode_length:
            action = robot.next_action_open_loop() 
            open_loop_list_of_states.append(robot.state)
            open_loop_list_of_actions.append(action)
            robot.state = environment.dynamics(robot.state, action)
        else:
            path = graphics.Path(open_loop_list_of_states, (0,0,255), 2, 0)
            if path not in robot.paths_to_draw:
                print('path not in')
                robot.paths_to_draw.append(path)

        if robot.open_loop_done == True:
            print('done')
            return True
        else:
            print('not done')
            return False
        
def run_closed_loop_planning():

    if len(closed_loop_list_of_states) < robot.episode_length:
        action = robot.next_action_closed_loop() 
        closed_loop_list_of_states.append(robot.state)
        closed_loop_list_of_actions.append(action)
        robot.state = environment.dynamics(robot.state, action)
    else:
        
        path = graphics.Path(closed_loop_list_of_states, (0,255,0), 2, 0)
        if path not in robot.paths_to_draw:
            print('path not in')
            robot.paths_to_draw.append(path)
    
    print('length ', len(closed_loop_list_of_states))
    if robot.close_loop_done == True:
        print('done closed')
        return True
    print('not done closed')
    return False


# Define what happens on each timestep.
def update(dt): 
    var = run_open_loop_planning()
    if var == True:
        run_closed_loop_planning()    
    #pyglet.app.exit()

    
    
#path = graphics.Path(list_of_states, (0,0,255), 2, 0)

#robot.paths_to_draw.append(path)

# Set how frequently the update function is called.
pyglet.clock.schedule_interval(update, 1/100)


# Finally, call the main pyglet event loop.
# This will continually update and render the environment in a loop.
pyglet.app.run()

