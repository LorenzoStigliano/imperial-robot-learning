### Analytical Methods

#### Introduction to models

- A model is a prediction of the next state $s'$ given the current state $s$ and action $a$![[Screenshot 2023-01-24 at 15.56.14.png]]
- There is always a true underlying model: $s' = g(s,a)$ which is determined by the laws of physics
- We analyse the system and create our own representation of this model $s' = f(s,a)$
- closer $f$ is to $g$ the more useful the model is to predict future states

Examples of known models:
- Games and discrete environments, we know exactly what happens given a state and an action we know what will happen

Examples of unknown models:
- Continuous space and action spaces very difficult to learn the model

Semi-known models: 
- We can get good enough estimates of the model environment 

**Moravec's paradox**
- Things that are easy for humans are very difficult for AI, and the counter is true. So if we know a model for the environment then easy to *solve*.

#### Model-based vs model-free learning

Model: $s' = f(s,a)$, Policy: $a=\pi(s)$
- We can use a model to learn a policy - model-based
- We can learn the policy without a model - model-free learning

#### Analytical states and models

There are two ways of learning top is analytical bottom is model-based RL
![[Screenshot 2023-01-24 at 16.07.12.png]]
- $o$ - observation
- $s$ - states that the robot can be in
- $a$ - action to take

#### Analytical state estimation - given an observation can we estimate a state
Example: 
- Observation: Joint encoder, so robot knows the position of the arm 
- Analysis: we can do analysis to determine the gripper
- Estimated state: position / orientation of gripper

#### Analytical models - being able to predict the next state given the current state and the action analytically
Example: 
- State: Position / Encoder of gripper
- Action: Motor torques
- Analysis: Mathematics Geometry
- Next State: Position / Encoder of gripper

#### Analytical control/planning
- Planning is the process of calculating a sequence of actions to optimise some objective function
- Usually, the objective function is the sum of rewards along the action sequence, $\sum_{t=1}^{t=T} r(s_t,a_t)$
- Understood as a robot “thinking ahead”, using its model of the world, to determine what actions to take

#### Planning with gradients

- Differentiable models: If $s'=f(s,a)$ is differentiable we can calculate how much to change the action, in order to optimise the next state
- Then we can do gradient based methods to optimize the function
![[Screenshot 2023-01-24 at 17.04.25.png]]
- We can backpropagate this backwards to optimise the action at each time step

#### Gradient-based optimisation is local PROBLEM
Problem:
- Planning with gradients is a local optimisation
	- If the optimisation landscape is non-convex, optimisation will result in a local optimum
Solution: 
- Planning with sampling
![[Screenshot 2023-01-24 at 17.08.12.png]]

#### Planning with Sampling

##### Techniques

1. Random Shooting
	- Sample $N$ action sequences $A_1, ..., A_N$ from distribution $p(A)$ where $A = a_1, .... a_T$
	- Simulate each sequence using model, $s_{t+1} = f(s_t,a_t)$
	- Score each sequence using reward function, $J(A) = \sum r(s_t, a_t)$
	- Execute sequence with highest score $A* = max J(A_n)$
- Advantages: 
	- Very simple to implement
	- Easy to parallelise
- Disadvantage:
	- Huge computational cost
	- Scales poorly with state and action dim

2. Cross entropy method
	- Loop:
		 1. Sample $N$ action sequences $A_1, ..., A_N$ from distribution $p(A)$ where $A = a_1, .... a_T$
		 2. Simulate each sequence using model, $s_{t+1} = f(s_t,a_t)$
		 3. Score each sequence using reward function, $J(A) = \sum r(s_t, a_t)$
		 4. Refit $p(A)$ to top $k$ % of $A_1, ..., A_N$, so we only sample from the best action sequence space
	 - Execute mean of $p(A)$ once we have a good set of solutions, every time step has a mean and variance 
- We simulate this and the at the end we execute
- Problem: Open-loop planning is usually **sub-optimal**![[Screenshot 2023-01-24 at 17.21.01.png]]

#### Planning under uncertainty

What if the model has error: $f(s,a)=g(s,a)+\epsilon$
- Dynamics may be stochastic
- State modelling may be imperfect
- Dynamics modelling may be imperfect or too complex to model

Closed-loop planning: model predictive control
- Loop: 
		1. Choose optimal action sequence (e.g. with cross entropy method) 
		2. Execute *first action* in this sequence 
		3. Re-plan entire sequence from current state onwards (1.)
![[Screenshot 2023-01-24 at 17.23.27.png]]
![[Screenshot 2023-01-24 at 17.23.06.png]]

Problems:
- Closed-loop planning still suffers from error sources
- We can learn the model for the dynamics