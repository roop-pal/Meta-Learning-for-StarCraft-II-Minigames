
This folder contains scripts that will setup "manually" an environemnt.

To build an Agent, you'd normally create and instantiate a subclass of the class BaseAgent from pysc2.agents.base_agent, for example MyOwnAgent, and then run the following to have the Agent running on a given map:

```
python -m pysc2.bin.agent --map Simple64 --agent test_scripted_agent.MyOwnAgent
```

This sets up an envrionment and makes the agent interact with the environment through the agent's step() function. Basically, it iteratively calls MyOwnAgent.step(obs) which return the action chosen by the agent, and then passes this to env.step() to perform the action.

In this folder, we do this "by hand", which might be easier to work with if we want to do more "lower level" things. This might help to link the neural networks and the game environment.


Note that you can do everything from ipython, which helps understanding pysc2 code, especially how the actions need to be declared.