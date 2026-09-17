import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

# neural network used by the agent to estimate Q values
class DQN(nn.Module):  

    # input (11) -> hidden 1 (128) -> hidden 2 (128) -> output (3)
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    # pass a sate through the network and return Q value
    def forward(self, state):
        return self.network(state)

# store previous experiences so that agent can train on randomly sampled experiences instead of only the most recent
class ReplayBuffer:  

    # deque removes oldest expeirnce when max capacity is reached
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # store one experience in replay buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    # randomly sample a batch of experiences
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    # return number of stored experiences
    def __len__(self):
        return len(self.buffer)

# AGENT
class Agent:
    def __init__(self):
        self.state_size=11       # danger(S,L,R) food(U,D,L,R) direction(U,D,L,R)
        self.action_size=3       # (S,L,R)
        self.gamma=0.99          # discount factor
        self.epsilon=1.0           # probability of exploration 
        self.epsilon_min=0.05    # min amount of exploration
        self.epsilon_decay=0.995 # reduction factor of episilon per iteration
        self.batch_size=64       # number of experiences used per training update

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # main network actively trained and used when selecting actions.
        self.model=DQN(self.state_size, self.action_size).to(self.device)

        # delayed copy of the main network used  to calculate the target Q value
        self.target_model=DQN(self.state_size,self.action_size).to(self.device)

        # experience replay memory
        self.memory=ReplayBuffer(100000)

        # start with identical weights on both networkks
        self.target_model.load_state_dict(self.model.state_dict())

        # optimizer used to update the main network
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001)

    # store an experience in replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # choose an action using epsilon greedy strategy
    def choose_action(self, state):

        # exploration
        if random.random() < self.epsilon: 
            return random.randrange(self.action_size)

        # exploitation
        state=torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values=self.model(state)
        return q_values.argmax(dim=1).item()

    # copy main network's weights into target network (called periodically during training instead of after every step)
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # trade exploration for exploitation
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    # save trained model's weights
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # load model weights on both main and target network
    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict) 
        self.target_model.load_state_dict(state_dict)

    # perform one DQN train step using a random batch from replay buffer
    def train(self):

        # dont start train until enough experiences exist to form one complete batch
        if len(self.memory) < self.batch_size:
            return

        # sample an experience
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states -> tensors
        states=torch.tensor(np.array(states),dtype=torch.float32).to(self.device)

        # actions -> integers to be used as indices when selecting Q(s,a)
        actions=torch.tensor(actions, dtype=torch.long).to(self.device)

        # rewards -> floats
        rewards=torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # net states -> tensors
        next_states=torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)

        # done -> binary (0.0: episode still running, 1.0: episode ended)
        dones=torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Q(s,a) produced for every action
        current_Q=self.model(states)
        current_Q=current_Q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # max(Q(s', a')) estimated from target network
        with torch.no_grad():
            next_Q=self.target_model(next_states)

            # choose max Q value among all possible next actions
            max_next_Q=next_Q.max(dim=1)[0]

            # BELLMAN EQUATION: target = reward + gamma*max(Q(s',a'))
            # if episode has ended: target = reward
            # (1-dones) removes Q(s',a') when done=1
            target_Q=rewards+(self.gamma*max_next_Q*(1-dones))

        # loss calculation by comparing predicted Q value against the target Q value
        loss=torch.nn.MSELoss()(current_Q, target_Q)

        # backpropagation (clear gradients from pervious step and update model weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()