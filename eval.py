from utils import preprocess_screen

import argparse
import gym
import learner
import numpy as np
import random
import sys
import torch


def eval(
    target_net, env, h, w, num_stacked, num_actions,
    num_episodes=1, max_steps_per_ep=1000, eps=0.01,
    render=False, verbose=False
    ):
    target_net.eval()
    rewards = [0] * num_episodes

    for ep in range(num_episodes):
        episode_reward = 0
        screen = env.reset()
        if (len(screen.shape) != 3):
            screen = env.render(mode="rgb_array")
        screen = torch.as_tensor(preprocess_screen(screen, h, w))

        if render:
            env.render(mode="human")

        # Init first state by duplicating initial screen
        cur_state = torch.zeros(
            [1, num_stacked, h, w], dtype=torch.float32)
        for i in range(num_stacked):
            cur_state[0][i] = screen

        for step in range(max_steps_per_ep):
            # Epsilon-greedy action selection
            if random.random() > eps:
                action = target_net(cur_state)[0].argmax().item()
            else:
                action = env.action_space.sample()

            screen, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break

            if len(screen.shape) != 3:
                screen = env.render(mode="rgb_array")
            screen = torch.as_tensor(preprocess_screen(screen, h, w))
            cur_state = torch.cat(
                (cur_state[:, 1:, ...], screen[None, None, :, :]), axis=1)

            if render:
                env.render(mode="human")

        if verbose:
            print("Episode {0}: Score = {1}".format(i + 1, episode_reward))

        rewards[ep] = episode_reward
        env.close()

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an actor/worker for generating experiences")
    parser.add_argument("-n", "--num_episodes", action="store", type=int, default=1,
                        help="Number of episodes")
    parser.add_argument("-e", "--epsilon", action="store", type=int, default=0.01,
                        help="Epsilon - probability of randomly acting")
    parser.add_argument("-p", "--path", action="store", required=True,
                        help="Path to the state dictionary of the model")
    parser.add_argument("-r", "--render", action="store_true",
                        help="Enable rendering to display.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable printing details about each episode.")
    args = parser.parse_args()

    game = "CartPole-v1"
    env = gym.make(game)
    h = w = 64
    num_stacked = 4
    num_actions = env.action_space.n
    max_steps_per_ep = 1000

    target_net = learner.DQN(h, w, num_stacked, num_actions)
    target_net.load_state_dict(torch.load(args.path))
    rewards = eval(
        target_net, env, h, w, num_stacked, num_actions,
        args.num_episodes, max_steps_per_ep, args.epsilon,
        args.render, args.verbose)

    print("Average score: {0}".format(np.mean(rewards)))
    print("Median score: {0}".format(np.median(rewards)))
    print("Standard deviation: {0}".format(np.std(rewards)))
    print("Min score: {0} | Max score: {1}".format(min(rewards), max(rewards)))
