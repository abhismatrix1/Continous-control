import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1=nn.Linear(state_size,400)
        self.fc2=nn.Linear(400+action_size,300)
        
        self.bn=nn.BatchNorm1d(state_size)
        self.bn2=nn.BatchNorm1d(400)
 
        self.fc5=nn.Linear(300,1)
        
        #last layer weight and bias initialization 
        torch.nn.init.uniform_(self.fc5.weight, a=-3e-4, b=3e-4)
        torch.nn.init.uniform_(self.fc5.bias, a=-3e-4, b=3e-4)
        
    def forward(self, state,action):
        """Build a network that maps state & action to action values."""
        
        x=self.bn(state)
        x=F.relu(self.bn2(self.fc1(x)))
        x=torch.cat([x,action],1)
        x=F.relu(self.fc2(x))
        
        x=self.fc5(x)
        
        return x

    
class Actor(nn.Module):

    def __init__(self,state_size, action_size, seed):
        super(Actor, self).__init__()
        

        # network mapping state to action 

        self.seed = torch.manual_seed(seed)
        
        self.bn=nn.BatchNorm1d(state_size)
        self.bn2=nn.BatchNorm1d(400)
        self.bn3=nn.BatchNorm1d(300)
        
        self.fc1= nn.Linear(state_size,400)
        self.fc2 = nn.Linear(400,300)
        self.fc4 = nn.Linear(300, action_size)
        
        #last layer weight and bias initialization 
        torch.nn.init.uniform_(self.fc4.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.fc4.bias, a=-3e-3, b=3e-3)
        
        # Tanh
        self.tan = nn.Tanh()
        
        
    def forward(self, x):

        x=self.bn(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))

        return self.tan(self.fc4(x))

    