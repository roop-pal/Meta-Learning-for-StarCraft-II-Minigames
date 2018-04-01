# Deep-Reinforcement-Learning-for-StarCraft-II-Battles

We intend to develop various reinforcement learning methods for two SC2LE minigames involving battles between groups of agents. This project was developed for the course COMS 4995 Deep Learning taught by Prof. Iddo Drori at Columbia University, in Spring 2018. Our team is composed by Connor Hargus, Jerome Kafrouni and Roop Pal who have contributed equally.

<div align="center">
  <a href="https://www.youtube.com/watch?v=-fKUyT14G-8"
     target="_blank">
    <img src="http://img.youtube.com/vi/-fKUyT14G-8/0.jpg"
         alt="DeepMind open source PySC2 toolset for Starcraft II"
         width="240" height="180" border="10" />
  </a>
  <a href="https://www.youtube.com/watch?v=6L448yg0Sm0"
     target="_blank">
    <img src="http://img.youtube.com/vi/6L448yg0Sm0/0.jpg"
         alt="StarCraft II 'mini games' for AI research"
         width="240" height="180" border="10" />
  </a>
  <a href="https://www.youtube.com/watch?v=WEOzide5XFc"
     target="_blank">
    <img src="http://img.youtube.com/vi/WEOzide5XFc/0.jpg"
         alt="Trained and untrained agents play StarCraft II 'mini-game'"
         width="240" height="180" border="10" />
  </a>
</div>

## Intro

Deep reinforcement learning has made significant strides in recent years, with results achieved in board games such as Go. However, there are a number of obstacles preventing such methods from being applied to more real-world situations. For instance, more realistic strategic situations often involve much larger spaces of possible states and actions, an environment state which is only partially observed, multiple agents to control, and a necessity for long-term strategies involving not hundreds but thousands or tens of thousands of steps. It has thus been suggested that creating learning algorithms which outperform humans in playing real-time strategy (RTS) video games would signal a more generalizable result about the ability of a computer to make decisions in the real world.

Of the current RTS games on the market, StarCraft II is one of the most popular. The recent release by Google’s DeepMind of SC2LE (StarCraft II Learning Environment) presents an interface with which to train deep reinforcement learners to compete in the game, both in smaller “minigames” and on full matches. The SC2LE environment is described on [DeepMind's github repo](https://github.com/deepmind/pysc2) 

In this project, we focus on solving two types of minigames, DefeatRoaches and DefeatZerglingsAndBanelines. In each case the player is given a set of units which it must use to defeat a set of enemy units. Each time the set of enemy units is defeated, the player is given ad- ditional units. A reward of +5 or +10 (depending on the minigame) is given for defeating each enemy unit and a re- ward (penalty) of -1 is given for each of the player’s units which is lost.

## Our work

We first implemented and used "baseline" agents that will let us evaluate more complex Reinforcement Learning agents. We will compare our results with "random" agents that choose any random action at each step, and scripted agents that intend to solve the minigame with a simple deterministic policy. The scripted agents can be found in the folder *scripted_agents*.

We then implemented a "smarter" baseline agent using a Q-table. For this to be possible, we reduced the action space to a few basic actions (mainly selecting units and attacking points), and also reduced the state space (a 4 by 4 grid indicating where the roaches are along with the number of marines left).

We then made a review of the current architectures used to solve these minigames. In their paper, DeepMind use the A3C algorithm (Asynchronous Advantage Actor Critic) with several architectures (*Atari-Net*, *FullyConv*, *FullyConv LSTM*) that are described in [section 4.3](https://deepmind.com/documents/110/sc2le.pdf) of the SC2LE paper. DeepMind did not include open source implementations of the architectures used in their paper, yet a few research team shared implementations, and our work relies on theirs. Useful github resources can be found in the *readme* of the *docs* folder of this repo. Of course, at the end of our project, we will update our *References* section in this readme to give full credit to other teams' work. All agents based on different reinforcement learning ideas (DQN, A3C...) will be in the *rl_agents* folder.

Our ultimate goal is to formalize the problem in a multi-agent setting rather than a single-agent setting. We will then have multiple agents, each controlling a unit of the game, following their own policies learned with a common Q value estimator. Our goal is to implement a multi agent algorithm similar to the work done by Foerster et al. in their paper "Counterfactual Multi-Agent Policy Gradients" (COMA). In their paper, they implemented COMA on StarCraft using the original raw API from Blizzard. For our work, we plan to use the pysc2 API (SC2LE) which truly emulates the way players actually play the game: in particular, agents cannot control units directly and simultaneously, since real players cannot (a real player has to select units before giving orders...). Therefore, technical issues arise and make the implementation less straightforward, nevertheless we suspect that being able to apply a multi-agent technique will let us get significant improvements.

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

**TODO** after we clean up the repo a little bit.

## Notes

**This is a preliminary version of our project readme, which will be updated to contain descriptions of all the methods used, a full description of the problem setting (size of the action-state space, assumptions made...) and of course our results.**

## References

This section will be updated at the end of our project to contain a full list of other research teams' work that we'll have used to complete our work.