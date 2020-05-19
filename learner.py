from collections import namedtuple
from eval import eval
from utils import preprocess_screen

import math
import numpy as np
import os
import random
import time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, mem):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = mem
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, num_stacked, outputs):
        super().__init__()

        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        # Layer params
        conv1_filters = 32
        conv1_size = 8
        conv1_stride = 4

        conv2_filters = 64
        conv2_size = 4
        conv2_stride = 2

        conv3_filters = 64
        conv3_size = 3
        conv3_stride = 1

        fc_size = 512

        # Calculate convolution output size for fc layer
        conv1_h = conv2d_size_out(h, conv1_size, conv1_stride)
        conv2_h = conv2d_size_out(conv1_h, conv2_size, conv2_stride)
        conv3_h = conv2d_size_out(conv2_h, conv3_size, conv3_stride)

        conv1_w = conv2d_size_out(w, conv1_size, conv1_stride)
        conv2_w = conv2d_size_out(conv1_w, conv2_size, conv2_stride)
        conv3_w = conv2d_size_out(conv2_w, conv3_size, conv3_stride)

        # Create layers
        self.conv1 = nn.Conv2d(num_stacked, conv1_filters, kernel_size=conv1_size, stride=conv1_stride)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv2_size, stride=conv2_stride)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=conv3_size, stride=conv3_stride)
        self.bn3 = nn.BatchNorm2d(conv3_filters)
        self.fc1 = nn.Linear(conv3_h * conv3_w * conv3_filters, fc_size)
        self.fc2 = nn.Linear(fc_size, outputs)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.relu(self.bn2(self.conv2(X)))
        X = F.relu(self.bn3(self.conv3(X)))
        X = F.relu(self.fc1(X.view(X.size(0), -1)))
        return self.fc2(X)


class Learner():

    def __init__(
        self, h, w, num_stacked, game, num_actions,
        batch_size=32, gamma=0.99,
        learning_rate=0.01, momentum=0.0, weight_decay=0.0,
        max_steps_per_ep = 1000, steps_per_target_update=500,
        steps_per_eval=500, steps_per_log=100,
        device="cpu", mem_capacity=100000, best_model_save_dir="tmp/"
        ):
        self.h = h
        self.w = w
        self.num_stacked = num_stacked
        self.game = game
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_steps_per_ep = max_steps_per_ep
        self.steps_per_target_update = steps_per_target_update
        self.steps_per_eval = steps_per_eval
        self.steps_per_log = steps_per_log
        self.device = device

        if best_model_save_dir[-1] != '/':
            best_model_save_dir += '/'
        self.best_model_save_dir = best_model_save_dir
        if not os.path.isdir(best_model_save_dir):
            os.mkdir(best_model_save_dir)

        self.env = gym.make(game)
        self.memory = ReplayMemory(mem_capacity)

        # Policy net for selecting action
        self.policy_net = DQN(h, w, num_stacked, num_actions).to(device)
        # Target net for evaluating action
        self.target_net = DQN(h, w, num_stacked, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Freeze weight

        # Backprop policy net only
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        self.best_eval_score = -math.inf

    def add_memories(self, memories):
        for mem in memories:
            self.memory.append(mem)

    def get_policy_net(self):
        return self.policy_net.state_dict()

    def learn(self, num_steps):
        while (len(self.memory) < self.batch_size):
            time.sleep(1)
        print("Learning started...")

        total_loss = 0
        device = self.device

        for step in range(1, num_steps + 1):
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.as_tensor(torch.cat(batch.state), device=device)
            action_batch = torch.as_tensor(torch.cat(batch.action), device=device)
            reward_batch = torch.as_tensor(torch.cat(batch.reward), device=device)

            # Calculate Q-value for current state with the given action
            state_action_Q = self.policy_net(state_batch).gather(1, action_batch)

            # Calculate expected Q-value for non-final next states only
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            non_final_next_states = torch.as_tensor(non_final_next_states, device=device)

            next_state_Q = torch.zeros(self.batch_size, device=device)
            next_state_Q[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0].detach()
            expected_state_action_Q = self.gamma * next_state_Q.unsqueeze(1) + reward_batch

            # Computer loss
            loss = F.mse_loss(state_action_Q, expected_state_action_Q)
            total_loss += loss.item()

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # Update target net periodically with policy net
            if step % self.steps_per_target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()

            if self.steps_per_log is not None and step % self.steps_per_log == 0:
                print("Step {0}: Avg. loss = {1} | Mem. capacity = {2}".format(
                    step, total_loss / self.steps_per_log, len(self.memory)))
                total_loss = 0

            if step % self.steps_per_eval == 0:
                print("Evaluating started...")
                rewards = eval(
                    self.target_net, self.env, self.h, self.w,
                    self.num_stacked, self.num_actions,
                    num_episodes=10, max_steps_per_ep=self.max_steps_per_ep, eps=0.01)
                avg_reward = np.mean(rewards)

                if avg_reward > self.best_eval_score:
                    print("New highscore: {0} | Previous highscore: {1}".format(
                        avg_reward, self.best_eval_score))
                    self.best_eval_score = avg_reward
                    torch.save(
                        self.target_net.state_dict(),
                        "{0}best_model_{1}.pt".format(
                            self.best_model_save_dir, self.game))
                else:
                    print("Score: {0} | Current highscore: {1}".format(
                        avg_reward, self.best_eval_score))

        print("Learning finished.")
