from collections import namedtuple
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
STEPS_PER_TARGET_UPDATE = 1000
STEPS_PER_LOG = 100
GAMMA = 0.99
OPT_LEARNING_RATE = 0.001
OPT_MOMENTUM = 0.95
OPT_WEIGHT_DECAY = 0.0

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
        self.memory[self.position] = Transition(**mem)
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

    def __init__(self, h, w, num_stacked, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        # Policy net for selecting action
        self.policy_net = DQN(h, w, num_stacked, num_actions).to(DEVICE)
        # Target net for evaluating action
        self.target_net = DQN(h, w, num_stacked, num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Freeze weight

        # Backprop policy net only
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=OPT_LEARNING_RATE, momentum=OPT_MOMENTUM, weight_decay=OPT_WEIGHT_DECAY)

    def add_memories(self, memories):
        for mem in memories:
            self.memory.append(mem)

    def get_policy_net(self):
        return self.policy_net.state_dict()

    def learn(self, num_steps):
        total_loss = 0
        while (len(self.memory) < BATCH_SIZE):
            time.sleep(1)
        print("Learning started...")

        for step in range(1, num_steps + 1):
            transitions = self.memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Calculate value for current state with the given action
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Calculate expected value for non-final next states only
            # Final states have value 0
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0].detach()
            expected_state_action_values = GAMMA * next_state_values.unsqueeze(1) + reward_batch

            # Computer loss
            loss = F.mse_loss(state_action_values, expected_state_action_values)
            total_loss += loss.item()

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # Update target net periodically with policy net
            if step % STEPS_PER_TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if step % STEPS_PER_LOG == 0:
                print("Step {0}: Avg. loss = {1} | Mem. capacity = {2}".format(
                    step, total_loss / STEPS_PER_LOG, len(self.memory)))
                total_loss = 0
