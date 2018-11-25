[image2]: https://github.com/abhismatrix1/Continous-control/blob/master/training_graph.png "training graph"

### Algorithm Implemented - DDPG ( 20 agents)
I have implemented DDPG algorithm for continous control environment (20 agents with its own copy of environment). Applied noise to actions for exploration. Have used noise as described in the paper. 

#### Update frequency
I updated the critic and actor model after every 20 times step 10 times in a go. In my case i found that traiining was stable as can be seen from below graph.

#### Chossen hyperparameters
- Replay buffer size = 1e6  
- Minibatch size = 64
- Discount factor = 0.99
- Soft update of target parameters coefficient(TAU) = 1e-3
- Actor network learning rate  = 1e-3
- Critic network learning rate = 1e-4
- How often to update the network (time step) = 20      


#### Actor artitecture 
Actor is a fully connected neural network with 2 hidden units with 400 and 300 neurons. Input to the netowrk is the state vector and output is the deterministic action values (continous space).


#### Critic artitecture 
Critic is again a fully connected network with 2 hidden units. It has two inputs- state and action vector. Action is feeded after first hidden layer. Its first hidden layer has 400 neurons and second hidden layer has 300 neurons. Output of the network is Q-Value for given input state and action value.


### Result of training
The environment was solved in 53 episodes. Average running scores graph is below
![Training Graph][image2]


### Future work

1. Implementation of adding noise to model parameter than to action will help the model to improve as shown by OpenAI.
