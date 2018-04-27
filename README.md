# Deep-Reinforcement-Learning-for-StarCraft-II-Battles

We intend to develop various reinforcement learning methods for five SC2LE minigames involving moving units to target locations as well as battles between groups of agents. This project was developed for the course COMS 4995 Deep Learning taught by Prof. Iddo Drori at Columbia University, in Spring 2018. This work is done by Connor Hargus, Jerome Kafrouni and Roop Pal who have contributed equally.

We started our project by partially reproducing the results obtained by DeepMind in their SC2LE publication, as shown by the table above. Then, we implemented a meta-learning strategy showing how an agent's skills can be transferred between minigames.

## Intro

Deep reinforcement learning has made significant strides in recent years, with results achieved in board games such as Go. However, there are a number of obstacles preventing such methods from being applied to more real-world situations. For instance, more realistic strategic situations often involve much larger spaces of possible states and actions, an environment state which is only partially observed, multiple agents to control, and a necessity for long-term strategies involving not hundreds but thousands or tens of thousands of steps. It has thus been suggested that creating learning algorithms which outperform humans in playing real-time strategy (RTS) video games would signal a more generalizable result about the ability of a computer to make decisions in the real world.

Of the current RTS games on the market, StarCraft II is one of the most popular. The recent release by Google’s DeepMind of SC2LE (StarCraft II Learning Environment) presents an interface with which to train deep reinforcement learners to compete in the game, both in smaller “minigames” and on full matches. The SC2LE environment is described on [DeepMind's github repo.](https://github.com/deepmind/pysc2) 

In this project, we focus on solving a variety of minigames, which capture various aspects of the full StarCraft II game. These minigames focus on tasks such as gathering resources, moving to waypoints, finding enemies, or skirmishing with units. In each case the player is given a homogeneous set of units (marines), and a reward is based off the minigame (+5 for defeating each enemy roach in DefeatRoaches, for example).

## Our work

We first implement and used"baseline" agents that will let us evaluate more complex reinforcement learning agents. We compare our results with "random" agents that choose any random action at each step, and simple scripted agents that intend to solve the minigame with a simple deterministic policy. The scripted agents can be found in the folder *scripted_agents*.

We then implemented a "smarter" baseline agent using a Q-table. For this to be possible, we reduced the action space to a few basic actions (mainly selecting units and attacking points), and also reduced the state space (a 4 by 4 grid indicating where the roaches are along with the number of marines left).

We then made a review of the current architectures used to solve these minigames. In their paper, DeepMind use the A3C algorithm (Asynchronous Advantage Actor Critic) with several architectures (*Atari-Net*, *FullyConv*, *FullyConv LSTM*) that are described in [section 4.3](https://deepmind.com/documents/110/sc2le.pdf) of the SC2LE paper. DeepMind did not include open source implementations of the architectures used in their paper, yet a few research teams shared implementations, and our work relies on theirs. Useful github resources can be found in the *readme* of the *docs* folder of this repo. All agents based on different reinforcement learning ideas (MLSH, A3C) will be in the *rl_agents* folder. Our A3C agent is mainly based on the work of [Xiaowei Hu](https://github.com/xhujoy) who provided an implementation of A3C for pysc2.

The main contribution is an implementation of a MLSH (Meta-Learning Shared Hierarchies) agent, which can be trained on multiple minigames, sharing sub-policies. A master policy selects which sub-policy to use given observations. This allows the agent to generalize to previously unseen minigames by just training a master policy. A more detailed explanation of the algorithm can be found in the [paper](#MLSH).

## Preliminary results

Results for 5 tractable minigames:

![alt text](./doc/table.PNG "Results Table")

We have successfully trained an A3C agent with AtariNet on 5 of the 7 minigames: MoveToBeacon, CollectMineralShards, FindAndDefeatZerglings, DefeatRoaches and DefeatZerglingsAndBanelings. We have also tried simpler approach: we wrote scripted bots to solve these games, and implemented a simple Q-Learning agent with simpler action and state spaces. We implemented a MLSH agent from scratch.

The videos below show (1) our A3C agent trained with Atarinet architecture, on 25,000 episodes, playing DefeatRoaches, (2) our simple Q-Learning agent trained on MoveToBeacon, and (3) our MLSH agent trained on 4 minigames, playing DefeatRoaches.

<div align="center">
  
  <a href="https://youtu.be/dEAh0g9SVS0"
     target="_blank">
    <img src="https://img.youtube.com/vi/dEAh0g9SVS0/0.jpg"
         alt="Trained A3C Atarinet agent playing DefeatRoaches"
         width="240" height="180" border="10" />
  </a>
  <a href="https://youtu.be/Z-H1QQKXbhQ"
     target="_blank">
     <img src="https://img.youtube.com/vi/Z-H1QQKXbhQ/0.jpg"
         alt="Trained A3C Atarinet agent playing DefeatRoaches"
         width="240" height="180" border="10" />
  </a>
   <a href="https://youtu.be/s5wGk7tql0c"
     target="_blank">
     <img src="https://img.youtube.com/vi/s5wGk7tql0c/0.jpg"
         alt="Trained MLSH Atarinet agent playing DefeatRoaches"
         width="240" height="180" border="10" />
  </a>
  
</div>

We find that the MLSH scores well to the previously unseen DefeatZerglingsAndBanelings minigame, though it unsurprisingly does not achieve the score of an agent trained on that single minigame. These results show the capabilities of the agent to generalize across minigames. Such an algorithm can be very powerful in developing a strong reinforcement learning agent playing the full game.

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

To run an agent, instead of calling pysc2 directly as in the instructions from DeepMind, run the main.py script of our project, with the agent class passed as a flag. For example, to run the q table agent:

```
$ python -m main --agent=rl_agents.qtable_agent.QTableAgent --map=DefeatRoaches
```

If no agent is specified, the A3C agent is run by default.

```
$ python -m main --map=DefeatRoaches
```

## References
1. [O. Vinyals, T. Ewalds, S. Bartunov, P. Georgiev. et al. StarCraft II: A New Challenge for Reinforcement Learning. Google DeepMind, 2017.](https://deepmind.com/documents/110/sc2le.pdf)
2. [V. Mnih, A. Badia, M. Mirza1, A. Graves, T. Harley, T. Lillicrap, D. Silver, K. Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning, 2016.](https://arxiv.org/pdf/1602.01783.pdf)
3. [K. Frans, J. Ho, X. Chen, P. Abbeel, J. Schulman. Meta Learning Shared Hierarchies. arXiv preprint arXiv:1710.09767v2, 2017.](https://arxiv.org/pdf/1710.09767.pdf)<a name="MLSH"></a>
