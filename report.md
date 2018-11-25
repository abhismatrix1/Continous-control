### Algorithm Implemented - DDPG ( 20 agents)
I have implemented DDPG algorithm for continous control environment (20 agents with its own copy of environment). Applied noise to actions for exploration. Have used noise as described in the paper. 

### Update frequency
I updated the critic and actor model after every 20 times step 10 times in a go. In my case i found that traiining was stable as can be seen from below graph.

### Result of training
The environment was solved in 53 episodes. Average running scores graph is below
![Training Graph](image2)

### improvement for future work

1. Implementation of adding noise to model parameter than to action will help the model to improve as shown by OpenAI.
2. 