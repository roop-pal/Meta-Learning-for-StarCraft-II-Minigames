from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def run_loop(agents, env, max_frames=0, mlsh=False, warmup=2, joint=8):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    # Store episode number for this thread

    num_test_runs = 50
    num_ep = 0

    while True:
      num_frames = 0
      timesteps = env.reset()

      # Note: agents is just a list of a single agent in this case
      for a in agents:
        a.reset()

        # Warm-up period where only master policy is trained of "warmup" episodes
        if mlsh and (num_ep % (warmup + joint + num_test_runs) < warmup):
          a.train_only_master = True
          a.test_run = False

          # Ended  joint training, reset master value and subpol choice parameters
          if (num_ep % (warmup + joint + num_test_runs)) == 0:
            print("Resetting master policy")
            a.reset_master()

          print('Warming Up...')

        # Joint training period where both master and subpolicies trained of "joint" episodes
        elif mlsh and (num_ep % (warmup + joint + num_test_runs) < warmup + joint):
          a.train_only_master = False
          a.test_run = False

          print('Joint Training...')

        # Do test runs after joint training has finished
        elif mlsh:
          a.test_run = True
          print('Testing...')

      num_ep += 1

      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = env.step(actions)

        # Done if exceeded max_frames and training run, or finished full minigame episode
        is_done = (num_frames >= max_frames and not agents[0].test_run) or timesteps[0].last()
        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
