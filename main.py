from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import importlib
import threading

from absl import app
from absl import flags
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
import tensorflow as tf

from run_loop import run_loop
from pysc2.env import run_loop as pysc2_run_loop

import numpy as np

COUNTER = 0
LOCK = threading.Lock()
FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "rl_agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 60, "Total agent steps.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

# Useful to choose number of subpolicies selected from by MLSH master controller
flags.DEFINE_integer("num_subpol", 2, "Number of subpolicies used for MLSH.")
flags.DEFINE_integer("subpol_steps", 10, "Number of subpolicies used for MLSH.")
# original flag not included by xhujoy but useful:
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("warmup_len", 100, "Number of episodes for warm up period of training master policy.")
flags.DEFINE_integer("joint_len", 500, "Number of episodes after warm up for training master and subpolicies.")

FLAGS(sys.argv)
if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']

LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net+'/'+FLAGS.agent.rsplit(".", 1)[-1]
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)

MLSH_TRAIN_MAPS = ["DefeatRoaches", "MoveToBeacon", "FindAndDefeatZerglings", "CollectMineralShards"]

def pysc2_run_thread(agent_cls, map_name, visualize):
  """Original version of run_thread used for most agents, from pysc2.bin.agent"""
  with sc2_env.SC2Env(
      map_name=map_name,
      agent_race=FLAGS.agent_race,
      bot_race=FLAGS.bot_race,
      difficulty=FLAGS.difficulty,
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
      minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agent = agent_cls()
    pysc2_run_loop.run_loop([agent], env, FLAGS.max_agent_steps)
    if FLAGS.save_replay:
      env.save_replay(agent_cls.__name__)


def run_thread(agent, map_name, visualize, mlsh=False):
  scores = list()
  with sc2_env.SC2Env(
    map_name=map_name,
    agent_race=FLAGS.agent_race,
    bot_race=FLAGS.bot_race,
    difficulty=FLAGS.difficulty,
    step_mul=FLAGS.step_mul,
    screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
    minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
    visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)

    init_time = time.time()

    replay_buffer = [] # will get observations of each step during an episode to learn once episode is done
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS, mlsh=mlsh, warmup=FLAGS.warmup_len, joint=FLAGS.joint_len):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          # end of an episode, agent has interacted with env and now we learn from the "replay"
          counter = 0
          with LOCK:
            global COUNTER
            COUNTER += 1
            counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          replay_buffer = []
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
          if COUNTER % 100 == 0:
            time_elapsed = round((time.time() - init_time) / 60, 2) # in minutes
            print('Total time elapsed: {} minutes, Average time per episode: {}'.format(time_elapsed, round(time_elapsed/COUNTER, 2)))
    #  elif is_done:
          obs = recorder[-1].observation
          score = obs["score_cumulative"][0]
          scores.append(score)
          print('(episode score: {}, mean score: {}, max score: {})'.format(score, np.mean(scores[-300:]), np.max(scores)))

    if FLAGS.save_replay:
      env.save_replay(agent.name)


def _main(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)

  if agent_name == "A3CAgent" or agent_name == "MLSHAgent":
    # these agents cannot be initiated similarly to classic agents
    agents = []
    for i in range(PARALLEL):
      if agent_name == "A3CAgent":
        agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
      else:  # i.e. MLSHAgent
        agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution, FLAGS.num_subpol, FLAGS.subpol_steps, i+1)

      agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
      agents.append(agent)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(LOG)
    for i in range(PARALLEL):
      agents[i].setup(sess, summary_writer)

    agent.initialize()
    if not FLAGS.training or FLAGS.continuation:
      global COUNTER
      COUNTER = agent.load_model(SNAPSHOT)

    mlsh = (agent_name == "MLSHAgent")

    # Run threads
    threads = []
    for i in range(PARALLEL - 1):
      if not mlsh:
        t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False, mlsh))
      else:  # i.e. MLSHAgent
        # Create agents on different minigames for each thread
        minigame = MLSH_TRAIN_MAPS[i % len(MLSH_TRAIN_MAPS)]
        print("\n Minigame for thread " + str(i + 1) + ": " + minigame + "\n")
        t = threading.Thread(target=run_thread, args=(agents[i], minigame, False, mlsh))

      threads.append(t)
      t.daemon = True
      t.start()
      time.sleep(5)

    minigame = MLSH_TRAIN_MAPS[(len(agents) - 1) % len(MLSH_TRAIN_MAPS)]
    print("\n Minigame for thread " + str(len(agents)) + ": " + minigame + "\n")
    run_thread(agents[-1], minigame, FLAGS.render, mlsh=mlsh)

    for t in threads:
      t.join()

    if FLAGS.profile:
      print(stopwatch.sw)
  
  else:
    # other agents just call the usual main loop from pysc2
    threads = []
    for _ in range(FLAGS.parallel - 1):
      t = threading.Thread(target=pysc2_run_thread, args=(agent_cls, FLAGS.map, False))
      threads.append(t)
      t.start()

    pysc2_run_thread(agent_cls, FLAGS.map, FLAGS.render)

    for t in threads:
      t.join()

    if FLAGS.profile:
      print(stopwatch.sw)


if __name__ == "__main__":
  app.run(_main)
