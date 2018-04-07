# Note: This code draws on code from tutorials aimed at the Simple64 minigame at:
# https://itnext.io/build-a-sparse-reward-pysc2-agent-a44e94ba5255

# A baseline agent using a Q-learning table lookup which gets updated progressively
# to better map states and actions to values. For simplicity in the action and
# state space, we quantize the grid to 4 by 4

# To Run:
#   Roaches: python -m pysc2.bin.agent --map DefeatRoaches --agent qtable_agent.QTableAgent --agent_race T --max_agent_steps 0 --norender
#   BanelingsAndZerglings: python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent qtable_agent.QTableAgent --agent_race T --max_agent_steps 0 --norender

import random
import math
import os.path
from time import sleep

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# Give names to common PySC2 functions and unit IDs for simplicity in later code.
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_MARINE = 48

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

# Specify a file in which to save the q-table of the trained agent
DATA_FILE = 'agent_save'

ACTION_DO_NOTHING = 'donothing'
ACTION_ATTACK_SINGLE = 'attack'
ACTION_ATTACK_ALL = 'attackall'

smart_actions = [
    ACTION_DO_NOTHING,
]

# Create actions for moving to any point on a 16 by 16 grid
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK_SINGLE + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))
            smart_actions.append(ACTION_ATTACK_ALL + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))


# Classs managing the Q table mapping state-action pairs to predicted value
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # Use greedy-epsilon to choose maximum q action epsilon of the time, random action otherwise
    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # Choose best predicted action
            state_action = self.q_table.ix[observation, :]

            # Randomly select from actions with same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # Choose action randomly
            action = np.random.choice(self.actions)

        return action

    # Takes previous state, previous action, reward received, and current state to update Q-table
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        # If this step is not final, update table using bootstrapped Q_table approximation
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # If final step, reward is simply observed reward r
        else:
            q_target = r

        # Update Q table to better reflect q_target
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)


    # Check that state has been seen before, if not append new state to q table
    def check_state_exist(self, state):

        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


# RL agent using Q table with quantized states and actions to choose actions
class QTableAgent(base_agent.BaseAgent):
    def __init__(self):
        super(QTableAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.scores = []

        # Each of our moves requires 2 steps, keep track of which step we're on in move_number
        self.move_number = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')


    # Splits an action word like 'attackall_3_2' into constituent pieces (action and location)
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)


    # Proceed one step in StarCraft time units
    # (note: two steps required for each of our actions to first select and then attack)
    def step(self, obs):
        super(QTableAgent, self).step(obs)

        # Uncomment sleep to slow down simulation and observe what agents are doing
        # sleep(0.5)

        # Case where episode is about to end, use observed final reward and update q table accordingly
        if obs.last():
            reward = obs.reward

            score = obs.observation['score_cumulative'][0]
            self.scores.append(score)
            print('Avg Score (prev. 500): ' + str(sum(self.scores[-500:])/min(len(self.scores), 500)))
            print('Max score (prev. 500): ' + str(max(self.scores[-500:])))

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None
            self.rewards = []

            self.move_number = 0

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        # First step trains from previous action, selects new action which performs select of one or all units
        if self.move_number == 0:
            self.move_number = 1

            # Quantize the current state to sixteen squares to reduce action space, also keep track of
            # the number of marines
            current_state = np.zeros(17)
            current_state[0] = obs.observation['player'][_ARMY_SUPPLY]

            # Make array of "hot squares" indicating locations of enemy (or neutral) units/beacons/shards
            hot_squares = np.zeros(16)
            enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            neutral_y, neutral_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_NEUTRAL).nonzero()

            enemy_y = np.concatenate((enemy_y, neutral_y))
            enemy_x = np.concatenate((enemy_x, neutral_x))
            # enemy_x += neutral_x

            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 16))
                x = int(math.ceil((enemy_x[i] + 1) / 16))

                hot_squares[((y - 1) * 4) + (x - 1)] = 1

            for i in range(0, 16):
                current_state[i + 1] = hot_squares[i]

            # print("Current state: ", current_state)

            reward = obs.reward

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

            rl_action = self.qlearn.choose_action(str(current_state))

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.splitAction(self.previous_action)

            # Randomly select a marine to attack with
            if smart_action == ACTION_ATTACK_SINGLE:
                unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # Select all marines to attack with
            elif smart_action == ACTION_ATTACK_ALL:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        # Second part of an action, i.e. the attack step
        elif self.move_number == 1:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_ATTACK_SINGLE or smart_action == ACTION_ATTACK_ALL:

                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED,
                                                                  (int(x) + (x_offset * 4), int(y) + (y_offset * 4))])

        return actions.FunctionCall(_NO_OP, [])