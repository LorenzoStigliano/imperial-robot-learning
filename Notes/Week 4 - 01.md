### Real-world Reinforcement Learning

#### 1. Why use RL?

- RL - Human provides feedback to robot for a specific task, on how good each action was, each time a robot tries an action
- Simple tasks - controller can be manually designed, no need for RL
- Moderate tasks - we can hand design the model and then use analytical methods to optimise the controller
	- can be challenging when the state estimation or dynamics model are complex
- Complex task - in continuous space RL is viable, define a reward function, and then enable the robot to learn a controller using reinforcement learning.

Sources of error in analytical controllers![[Screenshot 2023-01-30 at 11.28.01.png]]

#### 2. Reinforcement learning design choices

##### Reward function

1. Sparse rewards
	- $R(s,a) > 0$ only in certain situations 0 otherwise. Example: reach goal state 1 otherwise 0
	- Pros:
		- Simple to design
		- Theoretically will result in most optimal policy, due to minimal human bias in reward function
	- Cons: 
		- Can be very inefficient, and sometimes effectively impossible, to learn with, due to very low probability that random actions will result in task being completed
2. Dense rewards
	- $R(s,a)$ provided at every timestep and is proportional to how close task is to being completed
	- Pros: 
		- Reward for every action, so can start learning immediately 
	- Cons: 
		- Can be difficult to design a good dense reward function 
		- A poorly-designed dense reward function can result in optimal policy being sub-optimal for the task

##### Types of rewards

1. Analytical rewards
	- Reward provided by mathematical function written in code 
	- Suitable when reward function is simple
2. External rewards
	- Reward provided by human or some hardware setup
	- Suitable when reward function is complex and difficult to define mathematically
	- Usually used with sparse rewards

##### Designing the reset mechanism, how do we reset the robot so it can try again?
1. Static environment, that is the robot is fixed in a position
2. Constrained environment, no movement in the environment or very limited
3. Learning to rest, the robot does this on their own
4. Manually resetting, human rests the environment 

#### 3. Model-based reinforcement learning

##### DDPG Model-free RL![[Screenshot 2023-01-30 at 12.05.35.png]]
Model-free RL is inefficient in real world robotics

##### Model-based
![[Screenshot 2023-01-30 at 12.09.41.png]]
- Key difference we do not store a reward unlike model-free
- Note: if only external rewards (rather than analytical rewards) are available, we can learn the reward function in a similar way to learning the dynamics function

##### Generalisation in model-free RL
- Q-function: $Q(s,a)$
	- Generalising involves interpolating/extrapolating Q-values
	- But $Q(s,a)$ can change abruptly when task decisions become complex![[Screenshot 2023-01-30 at 12.14.39.png]]
- Model: $f(s,a)$
	- Generalising involves interpolating/extrapolating dynamics 
	- But  $f(s,a)$ can change abruptly where dynamics becomes complex

##### Model-based vs model-free reinforcement learning
Advantages of model-based over model-free 
- If dynamics model is a simpler function than policy or Q-value, model-based learning may generalise better than model-free learning 
- The same model can often be transferred across different tasks / different reward functions 
Disadvantages of model-based over model-free 
- If dynamics model is a more complex function than policy or Q-value, model-based learning may generalise worse than model free learning 
- Planning at each timestep is slower than simply executing a policy

Example
- Here, true Q-values change throughout the environment, with limited generalisation
- But if dynamics function is modelled correctly, dynamics can generalise well throughout the environment: $s' = s + f(s,a)$, this is the delta of how much you move given an action does generalise![[Screenshot 2023-01-30 at 12.19.49.png]]

Transferring the model
- Planning algorithm requires: 
	- Model $f_{\theta}(s,a)=s'$ and Reward function $R(s,a)=r$
- Given a new reward function, same model can be used for a new task
- Real-world physics knows nothing about the robot’s task

#### 4. Learning the model
- Models are trained with supervised learning![[Screenshot 2023-01-30 at 12.24.23.png]]![[Screenshot 2023-01-30 at 12.24.43.png]]

##### Linear vs non-linear models

- If g(s, a) (the true underlying model) is linear, generalisation will be good
- f(s, a) (the learned model) can then be linear ![[Screenshot 2023-01-30 at 12.26.16.png]]
- But if g(s, a) is non-linear, generalisation may be poor if f(s, a) is linear
- Here, f(s, a) should be non-linear![[Screenshot 2023-01-30 at 12.26.09.png]]

#### 5. Residual reinforcement learning
- Goal achieve sparse rewards more efficiently
- A hand-designed, analytical controller, often fails for complex tasks but often it can get the robot ~90% of the way
- Residual reinforcement learning combines analytical solutions with learning solutions, to get the best of both worlds
- By initialising with a good hand designed policy (the baseline), a small amount of exploration may lead to the sparse reward![[Screenshot 2023-01-30 at 12.31.21.png]]
- Model-free residual reinforcement learning![[Screenshot 2023-01-30 at 12.31.36.png]]
#### 6. Offline reinforcement learning
- Main idea we can reuse data from one robot to the next task to learn so we do not have to start from scratch for a new task ![[Screenshot 2023-01-30 at 12.37.36.png]]
- On-policy very data in-efficient 
- Offline data can come from any policy, this could come from another policy, random motion, human teleoperation
- If we know the analytical reward function $R(s,a)$ then we can also re-label old data with new rewards. We can re-label each $r$ then compute policy for new task 

##### Pros and cons of offline reinforcement learning
Pros 
- Collecting data is time-consuming and expensive 
- Collecting data autonomously is unsafe, whereas collecting data manually (e.g. human demonstrations or teleoperation) is much safer; this data can then be relabeled with any reward function 
Cons 
- Distribution shift: predictions are made from inferred data (e.g. Q-values with model-free learning, planned states in model-based learning), which cannot be verified with real-world data collection 
	- In regular reinforcement learning, robot’s current policy would be executed (with exploration noise) to collect more data
	- Therefore, data in buffer would approximate data in computed policy (either with model-free learning or model-based learning)
	- But in offline reinforcement learning, data in buffer is never updated and so policy is computed over “imagined” data![[Screenshot 2023-01-30 at 12.45.42.png]]
	- Solution: constraining policy
		- Additional penalty term to reward function: $d(s_t, D)$
		- Calculates distance between $s$ and nearest state in buffer $D$ (using a nearest neighbour search))
		- This penalises planning over states which are far away from the training data![[Screenshot 2023-01-30 at 12.47.28.png]]
- Example ![[Screenshot 2023-01-30 at 12.54.04.png]]
- This becomes particularly unstable with Q-learning, where Q-network bootstraps from previous predictions