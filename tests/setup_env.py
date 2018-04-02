# coding: utf-8

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from absl import app
from absl import flags

def setup_env(map_name, ):

   FLAGS = flags.FLAGS
   flags.DEFINE_bool("render", True, "Whether to render with pygame.")
   flags.DEFINE_integer("screen_resolution", 84,
                        "Resolution for screen feature layers.")
   flags.DEFINE_integer("minimap_resolution", 64,
                        "Resolution for minimap feature layers.")

   flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
   flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
   flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

   flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                       "Which agent to run")
   flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
   flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
   flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                     "Bot's strength.")

   flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
   flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
   flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

   flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

   flags.DEFINE_string("map", None, "Name of a map to use.")
   # flags.mark_flag_as_required("map")

   FLAGS(['nothing in there']) # parses the flags

   # map_name = 'Simple64'
   # map_name = 'DefeatRoaches'
   visualize = True

   env = sc2_env.SC2Env(
         map_name=map_name,
         agent_race=FLAGS.agent_race,
         bot_race=FLAGS.bot_race,
         difficulty=FLAGS.difficulty,
         step_mul=FLAGS.step_mul,
         game_steps_per_episode=FLAGS.game_steps_per_episode,
         screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
         minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
         visualize=visualize)

         
   env.reset()

   env = available_actions_printer.AvailableActionsPrinter(env)
   action_spec = env.action_spec()
   observation_spec = env.observation_spec()

   return env