
## Motivation for imitation learning

- We would like to not use spare rewards this is because it is very unlikely to obtain sparse reward through random exploration
- Hard to use dense rewards due to the fact that they are hard to design 
- Problems with rewards functions, how do people with no knowledge design a reward function?

- Two ways to provide demonstrations:
	- Kinesthetics teaching - move the robot physically
	- Third-person imitation learning - very difficult for robots to generalise
- Two ways to use demonstrations:
	- Learning how to preform a task - what actions to take
		- Behavioral Cloning
	- Learning what the task is - what the reward function is
		- Inverse Reinforcement Learning

## Behavioral Cloning
![[Screenshot 2023-02-06 at 11.34.42.png]]
- We are training a policy $\pi_{\theta}(s)=a$
- Main problem: state distribution problem![[Screenshot 2023-02-06 at 11.38.56.png]]
- Once the robot is outside of the training data the problem is that the robot will no longer know what to do 
- Solution: Diverse the training states so you have more exposure to more of the world so you can learn a good policy![[Screenshot 2023-02-06 at 11.40.11.png]]![[Screenshot 2023-02-06 at 11.40.31.png]]
## Interactive imitation learning

- Dataset Aggregation (DAgger)
	- We rollout policy, we start with a random policy
	- For every state that the robot visits we provide an action for each state, a demonstration to move towards the goal![[Screenshot 2023-02-06 at 11.59.16.png]]![[Screenshot 2023-02-06 at 11.59.33.png]]
	- We provide correction at every time step and each state has a label ![[Screenshot 2023-02-06 at 12.00.38.png]]
	- Main problem: This requires an action to be manually provided for every single state in a trajectory, which is very laborious. (3.)
- Uncertainty-based interactive imitation learning
	- After taking several steps, robot is now uncertain about what action to take, so **requests** demonstration blue
	- So then we provide a full demonstration ![[Screenshot 2023-02-06 at 12.04.53.png]]
	- ![[Screenshot 2023-02-06 at 12.05.15.png]]
	- How do we know when the robot needs a demonstration? 
	- Solution: Estimating uncertainty with ensembles
		- Training: 
			- Entire dataset of (state, action) pairs is split into subsets. 
			- Each network is trained on only one subset.
		- Inference: 
			- A state/observation is fed into each network. 
			- The mean and standard deviation is calculated across all network predictions. 
			- The robot executes the mean action. 
			- The standard deviation is the uncertainty.![[Screenshot 2023-02-06 at 12.08.17.png]]

## Hybrid imitation learning and reinforcement learning

- Human demonstration is usually suboptimal, example kicking a ball but the data, the example isn't the optimal way of kicking a ball.
- Pre-training and fine-tuning
- Step 1: Q-network with a replay buffer, this is demonstration data so already gives a targeted search 
- Step 2: Then collect further data through reinforcement learning to further optimise 1. But then throw away the demonstration data 
![[Screenshot 2023-02-06 at 12.18.13.png]]

Variation 1: Pre-loading the replay buffer:
-  Add demonstration data to replay buffer. Collect further data and combine with demonstration data in replay buffer.
- Agent avoids “forgetting” demonstration data.

Variation 2: Residual reinforcement learning![[Screenshot 2023-02-06 at 12.25.55.png]]
- We learn the *correction from the baseline policy*

## Inverse reinforcement learning

- Problem: Many demonstrations are required to cover state space 
- Solution: Infer human’s underlying reward function, then just use regular reinforcement learning![[Screenshot 2023-02-06 at 12.30.24.png]]
- Inverse RL is an Underspecified problem 
	- For any demonstration, there are infinitely many different reward functions which explain the behaviour
	- Which of these is the human’s intent?
- Solution: Principle of Maximum Entropy: *the probability distribution which best represents the current state of knowledge about a system is the one with largest entropy*
- If there are multiple reward functions which explain some behaviour, the one which induces a policy with the **largest entropy** is the most likely true reward function, because it makes minimal assumptions other than imitation
![[Screenshot 2023-02-06 at 12.36.04.png]]
## Generative Adversarial Networks (GANs)
- the generator loss creates better images
![[Screenshot 2023-02-06 at 12.38.28.png]]
- Generative Adversarial Imitation Learning
![[Screenshot 2023-02-06 at 12.38.43.png]]![[Screenshot 2023-02-06 at 12.41.26.png]]


