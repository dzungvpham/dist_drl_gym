## Distributed Deep Reinforcement Learning with PyTorch and OpenAI Gym

### Installation

First, install PyTorch (Python). The version used at the time of this project was 1.5.

Then, install OpenAI's Gym along with the Atari add-on.

### Run

First, run the leader/learner script with `python leader.py` on your desired machine.

Then, run the worker/actor script with `python worker.py` on your worker machines. Remember to set the address and port of the leader node correctkly.