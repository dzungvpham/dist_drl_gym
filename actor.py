from io import BytesIO
from learner import DQN
from utils import preprocess_screen

import gym
import numpy as np
import pickle
import random
import torch

DEVICE = "cpu"


class Actor():

    def __init__(self, h, w, num_stacked, num_actions, game, max_steps_per_ep, **kwargs):
        self.h = h
        self.w = w
        self.num_stacked = num_stacked
        self.num_actions = num_actions
        self.max_steps_per_ep = max_steps_per_ep

        self.policy_net = DQN(h, w, num_stacked, num_actions).to(DEVICE)
        self.policy_net.eval()
        self.env = gym.make(game)
        self.memory = []

    def update_policy_net(self, state_dict_binary):
        with BytesIO(state_dict_binary) as buff:
            state_dict = torch.load(buff, map_location=DEVICE)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()

    def act_one_episode(self, eps, frame_skip_min, frame_skip_max):
        num_stacked = self.num_stacked
        episode_reward = 0

        # Get initial screen
        env = self.env
        screen = env.reset()
        if (len(screen.shape) != 3):
            screen = env.render(mode="rgb_array")
        screen = preprocess_screen(screen, self.h, self.w)

        # Init first state by duplicating initial screen
        cur_state = np.zeros(
            [1, num_stacked, self.h, self.w], dtype=np.float32)
        for i in range(num_stacked):
            cur_state[0][i] = screen
        cur_state_bin = pickle.dumps(cur_state)

        # Start acting...
        for step in range(self.max_steps_per_ep):
            # Epsilon-greedy action selection
            if random.random() > eps:
                input = torch.from_numpy(cur_state)
                action = self.policy_net(input)[0].argmax().item()
            else:
                action = env.action_space.sample()

            # Apply same action to a random number of frames
            for i in range(random.randint(frame_skip_min, frame_skip_max)):
                screen, reward, done, _ = env.step(action)
                if len(screen.shape) != 3:
                    screen = env.render(mode="rgb_array")
                screen = preprocess_screen(screen, self.h, self.w)
                episode_reward += reward

                next_state = np.concatenate(
                    (cur_state[:, 1:, ...], screen[None, None, ...]), axis=1)
                next_state_bin = pickle.dumps(next_state) if not done else None

                # Add transition to memory
                self.memory.append({
                    "state": cur_state_bin,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state_bin
                })

                if done:
                    break

                cur_state = next_state
                cur_state_bin = next_state_bin

            if done:
                break

        env.close()

        return episode_reward
