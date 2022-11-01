import numpy as np
from env import OW
from torch.distributions import Categorical
env = OW()


import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Args:
            input_dim (int): state dimension.
            output_dim (int): number of actions.
            hidden_dim (int): hidden layer dimension (fully connected layer)
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

    def forward(self, state):
        """
        Returns a Q value
        Args:
            state (torch.Tensor): state, 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)
        """
        x = F.relu(self.linear1(state))
        x = self.linear3(x)
        return x


"""
Agent class that implements the DQN algorithm
"""
class DQN:
    def __init__(self,input_dim, output_dim, hidden_dim,learning_rate,batch_size, seed=None):
        self.output_dim = output_dim
        self.input_dim = input_dim  # Output dimension of Q network, i.e., the number of possible actions
        self.dqn = QNetwork(self.input_dim, self.output_dim, hidden_dim)  # Q network
        self.dqn_target = QNetwork(self.input_dim, self.output_dim, hidden_dim)  # Target Q network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.batch_size = batch_size  # Batch size
        self.gamma = 0.99  # Discount factor
        self.eps = 1.0  # epsilon-greedy for exploration
        self.loss_fn = torch.nn.MSELoss()  # loss function
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=learning_rate)  # optimizer for training # replay buffer
        self.lr = learning_rate
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.device = torch.device("cpu")
    
    def sample(self, state):
        x_t = self.dqn(state)
        action_prob = torch.softmax(x_t,dim=-1)
        action_distribution = Categorical(action_prob)  # this creates a distribution to sample from
        max_probability_action = torch.argmax(action_prob, dim=-1)


        action = action_distribution.sample().cpu()
        z = action_prob == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_prob + z)
        return action, (action_prob, log_action_probabilities), max_probability_action
    
    def select_action(self,state,time,a_,eps):
        s_tensor = torch.tensor(state).to(self.device)
        #one_hot encoding
        s_onehot = F.one_hot(s_tensor-1, num_classes = 13)

        t_tensor = torch.tensor(time).to(self.device)
        #one_hot encoding
        

        state_a = torch.FloatTensor(a_).to(self.device)
        #print([s_onehot,t_tensor.view(-1,1),state_a])
        state = torch.cat([s_onehot,t_tensor.view(-1,1),state_a],1).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            _, z, action = self.sample(state)
        
        action = action.detach().cpu().numpy().squeeze()
        for i in range(len(time)):
            prob = self.rng.random()
            if prob <= eps:
                action[i] = np.random.randint(3)
        return action,z

    def train(self,args, memory,μ):
        """
        Train the Q network
        Args:
            s0: current state, a numpy array with size 4
            a0: current action, 0 or 1
            r: reward
            s1: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """
        
        
        """
        state_batch: torch.Tensor with shape (self.batch_size, 4), a mini-batch of current states
        action_batch: torch.Tensor with shape (self.batch_size, 1), a mini-batch of current actions
        reward_batch: torch.Tensor with shape (self.batch_size, 1), a mini-batch of rewards
        next_state_batch: torch.Tensor with shape (self.batch_size, 4), a mini-batch of next states
        done_list: torch.Tensor with shape (self.batch_size, 1), a mini-batch of 0-1 integers, 
                   where 1 means the episode terminates for that sample;
                         0 means the episode does not terminate for that sample.
        """
        S, A, R, N_S, A_,T,T_,N_A_,agent_n,D = memory.sample(args.batch_size)
        state_batch_ = torch.from_numpy(S).to(self.device)
        time_batch = torch.from_numpy(T).float()
        action_batch = torch.from_numpy(A).int()
        reward_batch = torch.from_numpy(R).float()
        next_state_batch_ = torch.from_numpy(N_S).to(self.device)
        mean_action_batch = torch.from_numpy(A_).float()
        done_list = torch.from_numpy(D).float()
        next_time_batch = torch.from_numpy(T_).float()
        next_mean_action_batch = torch.from_numpy(N_A_).float()

        s_onehot = F.one_hot(state_batch_-1, num_classes = 13)
        state_batch = torch.cat([s_onehot,time_batch.view(-1,1),mean_action_batch],1).to(self.device)

        ns_onehot = F.one_hot(next_state_batch_-1, num_classes = 13)
        next_state_batch = torch.cat([ns_onehot,next_time_batch.view(-1,1),next_mean_action_batch],1).to(self.device)
        
        """
        Hint: You may use the above tensors: state_batch, action_batch, reward_batch, next_state_batch, done_list
              You may use self.dqn_target as your target Q network
              You may use self.loss_fn (or torch.nn.MSELoss()) as your loss function
              You may use self.optim as your optimizer for training the Q network
        """
        state_action_values = self.dqn(state_batch).gather(1, action_batch.long().view(-1,1))
        next_state_values = self.dqn_target(next_state_batch).max(1)[0].view(-1)
        target_values = (reward_batch + self.gamma * next_state_values*(1 - done_list)).view(-1,1)
        loss = self.loss_fn(state_action_values, target_values)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        #update_μ
        optimal_policy = np.copy(μ)
        z,_ = self.select_action(S,T,A_,args.eps)
        optimal_policy[agent_n,S-1,T] = z
        return optimal_policy

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        """
        Add samples to replay memory
        Args:
            state: current state, a numpy array with size 4
            action: current action, 0 or 1
            reward: reward
            next_state: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        # Decay epsilon
        if self.eps >= 0.01:
            self.eps *= 0.95
        self.lr = self.lr*0.99
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)
    
    def target_update(self):
        # Update the target Q network (self.dqn_target) using the original Q network (self.dqn)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
