3 main fields of applications:

- Robot navigation
	- E.g. self-driving cars, drones 
- Legged robots 
	- E.g. quadrupeds, bipedals 
- Robot manipulation 
	- E.g. robot arms and hands

## Challenge 1: task diversity

The more tasks a robot needs to learn, the more difficult it is to:
- Problem from robot:
	- Learn observation representation or policies that perform well across all tasks
- Problems for designers: 
	- Manually design observation representations that cover all possible observations
	- Manually design reward functions or provide demonstrations for all necessary tasks

## Challenge 2: dynamics complexity

The more complex the environment dynamics, the more difficult it is to:
- Problem from robot:
	- Learn the model in a model-based reinforcement learning method 
	- Learn with model-free reinforcement learning, as complex dynamics often also means complex Q-functions
- Problems for designers: 
	- Manually design analytical controllers

![[Screenshot 2023-02-20 at 11.35.03.png]]

## Robot navigation

- Task: Move from A to B, whilst obeying rules of the road 
- Observations: Cameras (e.g. RGB, depth), Lidar, proximity sensors 
- Actions: Engine power, brake power, steering wheel angle![[Screenshot 2023-02-20 at 11.42.13.png]]

### Case study: conditional imitation learning

- Behavioural cloning at a T-junction
	- We want to solve the problem of getting onto the T-junction
	- We have the demonstration data
	- But the average will be predicted with MSE loss so we see this policy rollout
	- As a result, for any position their are many actions the robot can take![[Screenshot 2023-02-20 at 11.47.43.png]]
- Solution: Conditioning on both state and context (what is the task, it will take from that particular state)![[Screenshot 2023-02-20 at 11.49.36.png]]
 
 *Question: Why is behavioural cloning particularly suitable for training self-driving cars?*
 - Easy to collect data, we can collect all the cars and record their actions
 - Imitation learning is better than RL since we cannot have errors in self driving cars, extremely dangerous

## Legged robots

- Task: Keep walking forwards, without falling over 
- Observations: Joint angles, joint force-torque measurements, cameras 
- Actions: Joint torques, velocities, or positions![[Screenshot 2023-02-20 at 11.55.51.png]]

## Case study: sim-to-real transfer

- Naïve deployment of simulation policies usually fails, Trained with just one set of parameters (e.g. mass, friction coefficient, etc)
- Solution: randomise simulation parameters during training, Trained with randomised parameters (known as domain randomisation)
- Model-free reinforcement learning in simulation with domain randomisation![[Screenshot 2023-02-20 at 12.04.18.png]]
- Learning to Walk in Minutes Using Massively Parallel Deep RL

*Why is reinforcement learning particularly suitable for training legged robots?*
- We can train in parallel in a diversity of different tasks
- Reward function is fixed for all locomotion tasks 


## Robot manipulation
- Task: Pick up object or perform action with object 
- Observations: Joint angles, joint force-torque measurements, cameras 
- Actions: Joint torques, velocities, or positions![[Screenshot 2023-02-20 at 12.10.55.png]]
- Design choices? Where do we put the robots camera![[Screenshot 2023-02-20 at 12.11.07.png]]

