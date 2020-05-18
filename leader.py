from collections import namedtuple
from io import BytesIO
from learner import Learner, ReplayMemory, Transition, DEVICE
from threading import Thread
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import Binary

import gym
import numpy as np
import pickle
import torch

GAME_NAME = "CartPole-v1"
NUM_STEPS = 1000000
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_STACKED = 4
MAX_EPS = 1
MIN_EPS = 0.1
EPS_STEP = 0.001
NUM_EPISODES_PER_ACTOR = 1000
EPISODES_PER_POLICY_UPDATE = 5
EPISODES_PER_MEMORY_UPDATE = 5
FRAME_SKIP_MIN = 2
FRAME_SKIP_MAX = 4

def run_server(learner, num_actions):
    def get_actor_config():
        return {
            "h": INPUT_HEIGHT, "w": INPUT_WIDTH, "num_stacked": INPUT_STACKED,
            "num_actions": num_actions, "game": GAME_NAME,
            "max_eps": MAX_EPS, "min_eps": MIN_EPS, "eps_step": EPS_STEP,
            "num_episodes": NUM_EPISODES_PER_ACTOR,
            "episodes_per_policy_update": EPISODES_PER_POLICY_UPDATE,
            "episodes_per_memory_update": EPISODES_PER_MEMORY_UPDATE,
            "frame_skip_min": FRAME_SKIP_MIN,
            "frame_skip_max": FRAME_SKIP_MAX
        }

    def get_policy_net():
        with BytesIO() as buff:
            torch.save(learner.get_policy_net(), buff)
            return Binary(buff.getvalue())

    def add_memories(memories):
        for memory in memories:
            memory["state"] = torch.as_tensor(np.loads(memory["state"].data), device=DEVICE)
            memory["action"] = torch.tensor([[memory["action"]]], dtype=torch.long, device=DEVICE)
            memory["reward"] = torch.tensor([[memory["reward"]]], dtype=torch.float, device=DEVICE)
            if (memory["next_state"] is not None):
                memory["next_state"] = torch.as_tensor(np.loads(memory["next_state"].data), device=DEVICE)

        learner.add_memories(memories)
        return True

    with SimpleXMLRPCServer(("localhost", 8888,), logRequests=False) as server:
        server.register_function(get_actor_config)
        server.register_function(add_memories)
        server.register_function(get_policy_net)

        print("Server started...")
        server.serve_forever()

# =============== MAIN ===============

env = gym.make(GAME_NAME)
num_actions = env.action_space.n
learner = Learner(INPUT_HEIGHT, INPUT_WIDTH, INPUT_STACKED, num_actions)

server_thread = Thread(target=run_server, args=(learner, num_actions))
server_thread.setDaemon(True)  # make thread exit when main exit
server_thread.start()

learner.learn(NUM_STEPS)
