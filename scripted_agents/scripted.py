"""
This agent is the one proposed by DeepMind in pysc2.agents.scripted_agents.
We might develop a smarter (scripted) version for this agent.
It will serve as a baseline to compare our RL agents.
"""

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_MARINE = 48
_BANELING = 9
_ZERGLING = 105
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        return actions.FunctionCall(_NO_OP, [])
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        return actions.FunctionCall(_NO_OP, [])
      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

class DefeatRoaches(base_agent.BaseAgent):
  """
  Step 1: Select all units of the army
  Step 2: Detect roaches on the screen, attack the one with maximum y coordinate.
  """

  def step(self, obs):
    super(DefeatRoaches, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      if not roach_y.any():
        return actions.FunctionCall(_NO_OP, [])
      index = numpy.argmax(roach_y)
      target = [roach_x[index], roach_y[index]]
      return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    else:
      return actions.FunctionCall(_NO_OP, [])
  
class DefeatZerglingsAndBanelings(base_agent.BaseAgent):
  """
  Step 1: Select all units of the army
  Step 2: Detect banelings on the screen, attack the one with maximum y coordinate.
  """  
  def step(self, obs):
    super(DefeatZerglingsAndBanelings, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      unit_type = obs.observation["screen"][_UNIT_TYPE]
      baneling_y, baneling_x = (unit_type == _BANELING).nonzero()
      if not baneling_y.any():
        return actions.FunctionCall(_NO_OP, [])
      index = numpy.argmax(baneling_y)
      target = [baneling_x[index], baneling_y[index]]
      return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    else:
      return actions.FunctionCall(_NO_OP, [])
  
class FindAndDefeatZerglings(base_agent.BaseAgent):
  """
  Step 1: Select all units of the army
  Step 2: Detect zerglings on the screen, attack the one with maximum y coordinate.
  Step 3: If no zerglings, move to next point queued on screen
  """
  def __init__(self):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.obs_spec = None
    self.action_spec = None
    self.queue = []
    for i in range(50):
      for j in range(50):
        self.queue.append([i,j])
  
  def step(self, obs):
    super(FindAndDefeatZerglings, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      unit_type = obs.observation["screen"][_UNIT_TYPE]
      baneling_y, baneling_x = (unit_type == _BANELING).nonzero()
      if not baneling_y.any():
        selected = obs.observation["screen"][_SELECTED]
        # Get all pixels that our marine is in
        player_y, player_x = (selected == 1).nonzero()
        # average position of selected marine
        player = [int(player_x.mean()), int(player_y.mean())]
        if len(self.queue) == 0:
          return actions.FunctionCall(_NO_OP, []) 
        if numpy.linalg.norm(numpy.array(player) - numpy.array(self.queue[0]) < 1):
          self.queue = self.queue[1:]
        a = actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, self.queue[0]])
        return a
      index = numpy.argmax(baneling_y)
      target = [baneling_x[index], baneling_y[index]]
      return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    else:
      return actions.FunctionCall(_NO_OP, [])