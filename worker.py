from actor import Actor
from xmlrpc.client import ServerProxy

import pickle

DEFAULT_SERVER_ADDR = "localhost"
DEFAULT_SERVER_PORT = "8888"

addr = "http://{0}:{1}/".format(DEFAULT_SERVER_ADDR, DEFAULT_SERVER_PORT)

with ServerProxy(addr, allow_none=True) as proxy:
    config = proxy.get_actor_config()
    num_episodes = config["num_episodes"]
    episodes_per_policy_update = config["episodes_per_policy_update"]
    episodes_per_memory_update = config["episodes_per_memory_update"]
    eps = config["max_eps"]
    min_eps = config["min_eps"]
    eps_step = config["eps_step"]
    frame_skip_min = config["frame_skip_min"]
    frame_skip_max = config["frame_skip_max"]

    actor = Actor(**config)
    actor.update_policy_net(proxy.get_policy_net().data)

    for i in range(1, num_episodes + 1):
        reward = actor.act_one_episode(eps, frame_skip_min, frame_skip_max)
        print("Episode {0}: Total reward = {1} | eps = {2}".format(i, reward, eps))

        if i % episodes_per_policy_update == 0:
            actor.update_policy_net(proxy.get_policy_net().data)

        if i % episodes_per_memory_update == 0:
            proxy.add_memories(actor.memory)
            actor.memory.clear()

        eps = max(eps - eps_step, min_eps)

    if len(actor.memory) > 0:
        proxy.add_memories(actor.memory)
