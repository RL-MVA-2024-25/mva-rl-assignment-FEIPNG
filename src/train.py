from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV,evaluate_HIV_population
import os
import torch
from copy import deepcopy
import numpy as np
import random
import torch.nn as nn
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        device = torch.device('mps')
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return 


    def load(self):
        device = torch.device('mps')
        self.path = os.getcwd() + "/model.pt"
        self.model = self.myDQN(device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return 
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def myDQN(self, device):
        dim_state = env.observation_space.shape[0]
        nb_action = env.action_space.n 
        DQN = torch.nn.Sequential(
            nn.Linear(dim_state, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, nb_action)
        ).to(device)
        return DQN

    def train(self):
        device = torch.device('mps')
        print('Using device:', device)
        self.model = self.myDQN(device)
        self.model.apply(self.init_weights)
        self.target_model = deepcopy(self.model).to(device)
        self.gamma = 0.98
        self.batch_size = 800
        self.nb_actions = env.action_space.n
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        # epsilon_stop = 20000
        epsilon_update = 100
        update_target_freq = 400
        # epsilon_step = (epsilon_max-epsilon_min)/epsilon_stop
        self.decay_rate = 0.001
        epsilon = self.epsilon_max
        self.replaybuffer = ReplayBuffer(300000, device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        nb_gradient_steps = 3
        episode_return = []
        episode = 0
        step = 0
        previous_score = 0    
        previous_pop_score = 0
        previous_best_pop_score = 0
        state, _ = env.reset()
        while episode <  200:
            # update epsilon
            if step > epsilon_update:
                epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (np.exp(-self.decay_rate * step))
            
            # choose an action
            action = env.action_space.sample() if np.random.rand() < epsilon else self.act(state)

            # apply the action
            next_state, reward, done_step, trunc_step, _ = env.step(action)
            self.replaybuffer.add_buffer(state, action, reward, next_state, done_step)

            # calculate gradient and update model
            for _ in range(nb_gradient_steps):
                if len(self.replaybuffer) > self.batch_size:
                    # double DQN
                    state, action, reward, nextstate, done = self.replaybuffer.sample(self.batch_size)
                    Q_next_max = self.target_model(nextstate).max(1)[0].detach()
                    update = reward + self.gamma * Q_next_max * (1 - done)
                    Q_state_action = self.model(state).gather(1, action.to(torch.long).unsqueeze(1))
                    loss = self.criterion(Q_state_action, update.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)  
                    self.optimizer.step()

            # update target model
            if step % update_target_freq == 0: 
                self.target_model.load_state_dict(self.model.state_dict())

            # evaluate and choose best model
            if done_step or trunc_step:
                episode += 1
                eva_score = evaluate_HIV(agent=self, nb_episode=5)
                eva_pop_core = evaluate_HIV_population(agent=self, nb_episode=5)
                print(f"In episode {episode}, evaluation score : {eva_score:.2e},evaluation pop score : {eva_pop_core:.2e}")
                
                if eva_pop_core > previous_pop_score:
                    previous_pop_score = eva_pop_core
                    self.best_pop_model = deepcopy(self.model).to(device)
                    print("save model pop")
                
                if eva_score <= 2e10 :
                    if eva_score > previous_score:
                        previous_score = eva_score
                        previous_best_pop_score = eva_pop_core
                        self.best_model = deepcopy(self.model).to(device)
                        print("save model")
                else:
                    if eva_pop_core > previous_best_pop_score:
                        previous_best_pop_score = eva_pop_core
                        self.best_model = deepcopy(self.model).to(device)
                        print("save best model pop")
                state, _ = env.reset()
            else:
                state = next_state
            step += 1

        # save model
        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()+"/model.pt"
        self.save(path)
        self.model.load_state_dict(self.best_pop_model.state_dict())
        path = os.getcwd()+"/model_pop.pt"
        self.save(path)
        return episode_return

class ReplayBuffer:
    def __init__(self, max_size, device):
        self.max_size = max_size 
        self.index = 0
        self.data = []
        self.device = device

    def add_buffer(self, state, action, reward, next_state, done):
        if len(self.data) < self.max_size:
            self.data.append(None)
        self.data[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.max_size

    def sample(self, nb):
        # generate samples
        batch = random.sample(self.data, nb)        
        states, actions, rewards, next_states, dones = zip(*batch)
        # to tensor
        states = torch.Tensor(np.array(states)).to(self.device)
        actions = torch.Tensor(np.array(actions)).to(self.device)
        rewards = torch.Tensor(np.array(rewards)).to(self.device)
        next_states = torch.Tensor(np.array(next_states)).to(self.device)
        dones = torch.Tensor(np.array(dones)).to(self.device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.data)
    
# agent = ProjectAgent()
# agent.train()