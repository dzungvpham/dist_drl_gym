from utils import preprocess_screen

import learner
import gym
import random
import sys
import torch


def eval(
    target_net, env, h, w, num_stacked, num_actions,
    num_episodes=1, max_steps_per_ep=1000, eps=0.01
    ):
    target_net.eval()
    total_reward = 0

    for _ in range(num_episodes):
        screen = env.reset()
        if (len(screen.shape) != 3):
            screen = env.render(mode="rgb_array")
        screen = torch.as_tensor(preprocess_screen(screen, h, w))

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
            total_reward += reward
            if done:
                break

            if len(screen.shape) != 3:
                screen = env.render(mode="rgb_array")
            screen = torch.as_tensor(preprocess_screen(screen, h, w))
            cur_state = torch.cat(
                (cur_state[:, 1:, ...], screen[None, None, :, :]), axis=1)

        env.close()

    return total_reward / num_episodes


if __name__ == "__main__":
    game = "CartPole-v1"
    env = gym.make(game)
    h = w = 64
    num_stacked = 4
    num_actions = env.action_space.n
    num_episodes = 10
    max_steps_per_ep = 1000
    eps = 0.01

    target_net = learner.DQN(h, w, num_stacked, num_actions)
    target_net.load_state_dict(torch.load(sys.argv[1]))
    avg_reward = eval(target_net, env, h, w, num_stacked,
                      num_actions, num_episodes, max_steps_per_ep, eps)
    print("Average reward over {0} episode(s): {1}".format(
        num_episodes, avg_reward))
