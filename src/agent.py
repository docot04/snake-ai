import torch
import random
import numpy as np
from model import DQN, ReplayBuffer

class Agent:
    def __init__(self):
        self.state_size=11
        self.action_size=3
        self.gamma=0.99
        self.epsilon=1
        self.epsilon_min=0.05
        self.epsilon_decay=0.99
        self.batch_size=64
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model=DQN(self.state_size, self.action_size).to(self.device)
        self.target_model=DQN(self.state_size,self.action_size).to(self.device)
        self.memory=ReplayBuffer(100000)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001) #later

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def choose_action(self, state):
        # exploration
        if random.random() < self.epsilon: 
            return random.randrange(self.action_size)
        # exploitation
        state=torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values=self.model(state)
        return q_values.argmax(dim=1).item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path)) 
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states=torch.tensor(np.array(states),dtype=torch.float32).to(self.device)
        actions=torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards=torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states=torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones=torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        #Q(s,a)
        current_Q=self.model(states)
        current_Q=current_Q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # max(Q(s', a'))
        with torch.no_grad():
            next_Q=self.target_model(next_states)
            max_next_Q=next_Q.max(dim=1)[0]
            target_Q=rewards+(self.gamma*max_next_Q*(1-dones))

        loss=torch.nn.MSELoss()(current_Q. target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #trade exploration for exploitation

        if self.epsilon>self.epsilon_min:
            self.epsilon=self.epsilon * self.epsilon_decay