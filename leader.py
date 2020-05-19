from collections import namedtuple
from io import BytesIO
from learner import Learner, ReplayMemory, Transition
from socketserver import ThreadingMixIn
from threading import Thread
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import Binary

import argparse
import gym
import numpy as np
import pickle
import torch

# General config
GAME_NAME = "CartPole-v1"
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_STACKED = 4

# Actor config
MAX_EPS = 1
MIN_EPS = 0.1
EPS_STEP = 0.001
MAX_STEPS_PER_EP = 1000
NUM_EPISODES_PER_ACTOR = 1000
EPISODES_PER_UPDATE = 5
FRAME_SKIP_MIN = 2
FRAME_SKIP_MAX = 4

# Learner config
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_STEPS = 1000000
BATCH_SIZE = 32
MEMORY_CAPACITY = 100000
STEPS_PER_TARGET_UPDATE = 1000
STEPS_PER_LOG = 100
GAMMA = 0.99
LEARNING_RATE = 0.001
MOMENTUM = 0.95
WEIGHT_DECAY = 0.0


class MultiThreadedRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


def run_server(learner, num_actions, port):
    def get_actor_config():
        return {
            "h": INPUT_HEIGHT, "w": INPUT_WIDTH, "num_stacked": INPUT_STACKED,
            "num_actions": num_actions, "game": GAME_NAME,
            "max_eps": MAX_EPS, "min_eps": MIN_EPS, "eps_step": EPS_STEP,
            "max_steps_per_ep": MAX_STEPS_PER_EP,
            "num_episodes": NUM_EPISODES_PER_ACTOR,
            "episodes_per_update": EPISODES_PER_UPDATE,
            "frame_skip_min": FRAME_SKIP_MIN,
            "frame_skip_max": FRAME_SKIP_MAX
        }

    def get_policy_net():
        with BytesIO() as buff:
            torch.save(learner.get_policy_net(), buff)
            return Binary(buff.getvalue())

    def _convert_mem_to_transition(mem):
        next_state = mem["next_state"].data if mem["next_state"] != None else None
        return Transition(
            state=torch.as_tensor(np.loads(mem["state"].data)),
            action=torch.tensor([[mem["action"]]], dtype=torch.long),
            reward=torch.tensor([[mem["reward"]]], dtype=torch.float),
            next_state=torch.as_tensor(
                np.loads(next_state)) if next_state != None else None
        )

    def add_memories(memories):
        learner.add_memories(list(map(_convert_mem_to_transition, memories)))
        return True

    def add_memories_and_get_policy_net(memories):
        add_memories(memories)
        return get_policy_net()

    with MultiThreadedRPCServer(("localhost", port), logRequests=False) as server:
        server.register_function(get_actor_config)
        server.register_function(get_policy_net)
        server.register_function(add_memories)
        server.register_function(add_memories_and_get_policy_net)

        print("Server started...")
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a learner/leader for learning on localhost")
    parser.add_argument("--port", action="store", type=int, default=8888,
                        help="Port of leader machine. E.g: 8888")
    args = parser.parse_args()

    num_actions = gym.make(GAME_NAME).action_space.n
    learner = Learner(h=INPUT_HEIGHT, w=INPUT_WIDTH,
                      num_stacked=INPUT_STACKED, game=GAME_NAME, num_actions=num_actions)

    server_thread = Thread(target=run_server, args=(learner, num_actions, args.port))
    server_thread.setDaemon(True)  # make thread exit when main exit
    server_thread.start()

    learner.learn(NUM_STEPS)
