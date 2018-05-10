# Meta-Learning for StarCraft II Minigame Strategy
## Getting started

To get started, follow the instructions on the [pysc2 repository](https://github.com/deepmind/pysc2). As described in their instructions, make sure that the environment is set up correctly by running:

```
$ python -m pysc2.bin.agent --map Simple64
```

Our project relies on a few more packages, that can be installed by running:

```
$ pip install -r requirements.txt
```

We have tested our project using python 3 and pysc2 version 1.2, which is the main version currently available.

We are currently training our agents on a google cloud instance with a 4 core CPU and two Tesla K80 GPUs. This configuration might evolve during the project.

## Running agents

To run an agent, instead of calling pysc2 directly as in the instructions from DeepMind, run the main.py script of our project, with the agent class passed as a flag. For example, to run the q table agent or the MLSH agent:

```
$ python -m main --agent=rl_agents.qtable_agent.QTableAgent --map=DefeatRoaches
$ python -m main --agent=rl_agents.mlsh_agent.MLSHAgent --num_subpol=3 --subpol_steps=5 --training
```

If no agent is specified, the A3C agent is run by default:

```
$ python -m main --map=DefeatRoaches
```
A full list of the flags that can be used along with their descriptions is available in the main.py of script. The most important and useful flags are:

- map: the map on which to run the agent. Should not be used with MLSHAgent which uses a list of maps to use, since MLSH trains on multiple maps.
- max_agent_steps: the number of steps to perform per episode (after which, episode is stopped). This is used to speed up training by focusing on early states of episodes
- parallel: number of threads to run, defaults at 1.

Flags specific to the MLSHAgent:

- num_subpol: number of subpolicies to train and use
- subpol_steps: periodicity of subpolicy choices done by the master policy (in game steps)
- warmup_len: number of episodes during which only the master subpolicy is trained
- join_len: number of episodes during which both master and subpolicies are trained

## Acknowledgements

Our code is based on the work of Xiaowei Hu (xhujoy) who shared his implementation of A3C for pysc2.

Special thanks to Professor Iddo Drori, our instructor at Columbia University, as well as Niels Justesen for their expertise and guidance.

## References
1. [O. Vinyals, T. Ewalds, S. Bartunov, P. Georgiev. et al. StarCraft II: A New Challenge for Reinforcement Learning. Google DeepMind, 2017.](https://deepmind.com/documents/110/sc2le.pdf)
2. [V. Mnih, A. Badia, M. Mirza1, A. Graves, T. Harley, T. Lillicrap, D. Silver, K. Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning, 2016.](https://arxiv.org/pdf/1602.01783.pdf)
3. [K. Frans, J. Ho, X. Chen, P. Abbeel, J. Schulman. Meta Learning Shared Hierarchies. arXiv preprint arXiv:1710.09767v2, 2017.](https://arxiv.org/pdf/1710.09767.pdf)<a name="MLSH"></a>
4. [Xiaowei Hu's PySC2 Agents](https://github.com/xhujoy/pysc2-agents)

