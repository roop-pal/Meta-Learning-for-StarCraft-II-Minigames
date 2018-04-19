from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  avg_scores = []

  try:
    num_ep = 0
    while True:
      num_frames = 0
      timesteps = env.reset()

      # Note that agents is just a list of a single agent in this case
      for a in agents:
        a.reset()
        a.training = not ((num_ep % 30) == 0)
        print('Training: ' + str(a.training))

      num_ep += 1

      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = env.step(actions)
        # Only for a single player!
        is_done = (num_frames >= max_frames and agents[0].training) or timesteps[0].last()
        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
