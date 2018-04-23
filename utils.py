from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


def preprocess_minimap(minimap):
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    else:
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  # for i in range(len(features.SCREEN_FEATURES)):
  features_v11 = list(range(9)) + [10,12,14,15] # quickfix to make it compatible with pysc2 v1.2
  for i in features_v11:
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    else:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def minimap_channel():
  c = 0
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      c += 1
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.MINIMAP_FEATURES[i].scale
  return c


def screen_channel():
  c = 0
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      c += 1
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.SCREEN_FEATURES[i].scale
  return c

def preprocess_obs(obs, isize):
  minimap = np.array(obs.observation['minimap'], dtype=np.float32)
  minimap = np.expand_dims(preprocess_minimap(minimap), axis=0)

  screen = np.array(obs.observation['screen'], dtype=np.float32)
  screen = np.expand_dims(preprocess_screen(screen), axis=0)
  # TODO: only use available actions
  info = np.zeros([1, isize], dtype=np.float32)
  info[0, obs.observation['available_actions']] = 1
  return minimap, screen, info

def preprocess_rbs(rbs, isize):
  """
  Takes a (reversed) replay buffer rbs, and creates numpy arrays for minimaps, screens and infos
  to feed the network during update
  """
  minimaps = []
  screens = []
  infos = []

  # process the observations from the replay to use them for the update:
  for i, [obs, action, next_obs] in enumerate(rbs):
    minimap, screen, info = preprocess_obs(obs, isize)

    minimaps.append(minimap)
    screens.append(screen)
    infos.append(info)

  minimaps = np.concatenate(minimaps, axis=0)
  screens = np.concatenate(screens, axis=0)
  infos = np.concatenate(infos, axis=0)

  return minimaps, screens, infos
