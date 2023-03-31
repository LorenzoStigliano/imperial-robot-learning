
![[Screenshot 2023-02-13 at 11.55.27.png]]![[Screenshot 2023-02-13 at 11.55.33.png]]

## Learning representations of observations

- Observation as input to policy: high-dimensional, therefore more difficult to generalise 
- State as input to policy: low-dimensional, therefore generalises easily
- But state is not available! 
- Therefore, we try to learn a low-dimensional representation of the observation![[Screenshot 2023-02-13 at 12.12.03.png]]![[Screenshot 2023-02-13 at 12.13.37.png]]
- If we had labels of where the cup is, and trained to predict the position of the cub, this could learn a useful representation.
- Features which can predict where the cup is, would also be useful features to predict what the Q-value is, or the next state. 
- But this would be supervised, not unsupervised!

### Predicting the observation
- Instead, we can predict the observation itself, with an auto-encoder
![[Screenshot 2023-02-13 at 12.15.09.png]]

## Contrastive learning 

- The problem with auxiliary learning using auto-encoders
	- If this can reconstruct the observation, then it unnecessarily encodes the flowers, the candle, the shadow, etc.
	- Therefore, it is not a “minimal but sufficient” representation of the observation
	- Ideally, it should only encode the mug
	- But there is nothing telling the auto-encoder what is relevant and what is irrelevant
![[Screenshot 2023-02-13 at 12.20.58.png]]
![[Screenshot 2023-02-13 at 12.22.25.png]]
- Here we encode the same image using different colors so learning is invariant to the color
- We need to make sure it also tells the difference between objects thus we use NEGATIVES
- This will ensure that the observation representation is invariant to the augmentations, whilst being sensitive to everything else
- Therefore, this representation will learn what is important (e.g. position of object) and what is not important (e.g. illumination)![[Screenshot 2023-02-13 at 12.22.48.png]]

## Knowledge-based intrinsic motivation

- Rewards encourage learning of a specific task 
- Requires designing many reward functions, which can be tedious 
- Human equivalent – learning to play an instrument

Two ways to use intrinsic motivation:
- Reward-free pre-training with intrinsic rewards 
	- First, pre-train without task reward available (same pre-training for all tasks)
	- Then, use this knowledge for subsequent tasks when given task reward function
- Exploration with intrinsic rewards
	- First, explore until start receiving rewards (in sparse reward environment) 
	- Then, trade-off exploration and exploitation

### Two types of intrinsic reward

Knowledge-based 
- Reward given for high “surprise in environment behaviour”
Competence based 
- Reward given for high “ability to change the environment”

- Intrinsic rewards are used like regular rewards (e.g. in a model-free or model-based reinforcement learning framework) 
- But they encourage good actions to be taken without knowledge of the specific tasks that need to be learned

### Exploiting error in learned model

- If we have learned model ![[Screenshot 2023-02-13 at 12.38.46.png]]
- Then prediction error is a proxy for “surprise” 
- High surprise means the robot should revisit this state, and therefore should be assigned a high reward 
- Why is this a sensible intrinsic reward? 
	- First, this will reduce the model error more quickly 
	- Second, dynamics that are difficult to model suggests “interesting” region of state space

## Competency-based intrinsic motivation

- Empowerment: the level of “preparedness” to be able to achieve diverse things (reach diverse states) in the future.
- Formally, empowerment is an information theoretic measure: how much “information” can be sent from the actions the robot can take, to the states the robot can achieve.
- In other words, high empowerment means that a robot can intentionally reach diverse states, using its actions.

### Maximising mutual information
![[Screenshot 2023-02-13 at 12.48.28.png]]
- Diversity Is All You Need (DIAYN)