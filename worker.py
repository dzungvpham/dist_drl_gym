from actor import Actor
from xmlrpc.client import ServerProxy

import argparse
import pickle

parser = argparse.ArgumentParser(description="Create an actor/worker for generating experiences")
parser.add_argument("--host", action="store", default="localhost",
                    help="Address of leader machine. E.g: localhost")
parser.add_argument("--port", action="store", default="8888",
                    help="Port of leader machine. E.g: 8888")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable printing details about each episode.")
args = parser.parse_args()

addr = "http://{0}:{1}/".format(args.host, args.port)

try:
    with ServerProxy(addr, allow_none=True) as proxy:
        config = proxy.get_actor_config()
        num_episodes = config["num_episodes"]
        episodes_per_update = config["episodes_per_update"]
        eps = config["max_eps"]
        min_eps = config["min_eps"]
        eps_step = config["eps_step"]
        frame_skip_min = config["frame_skip_min"]
        frame_skip_max = config["frame_skip_max"]

        actor = Actor(**config)
        actor.update_policy_net(proxy.get_policy_net().data)

        for i in range(1, num_episodes + 1):
            reward = actor.act_one_episode(eps, frame_skip_min, frame_skip_max)
            if args.verbose:
                print("Episode {0}: Total reward = {1} | eps = {2}".format(i, reward, eps))

            if i % episodes_per_update == 0:
                policy_net = proxy.add_memories_and_get_policy_net(actor.memory)
                actor.update_policy_net(policy_net.data)
                actor.memory.clear()

            eps = max(eps - eps_step, min_eps)

        if len(actor.memory) > 0:
            proxy.add_memories(actor.memory)
except Exception:
    print("An exception occurred. Check the leader's address/port or restart the worker or leader.")
