import setup_env
import time
from pysc2.lib import actions

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id


env = setup_env.setup_env('DefeatRoaches')
# this object represents our interaction with the world
# to perform actions, we call env.step() with a list of actions
# don't forget to close the environment at the end with env.close()
# which shuts down the game 

# Examples:
# # moving the screen:
# env.step([actions.FunctionCall(1, [(20,20)])])
# # Selecting a rectangle:
# env.step([actions.FunctionCall(3, [(0,), (0,0), (80,80)])])
# # Selecting a unit among the ones you have already in your multi-select:
# env.step([actions.FunctionCall(5, [(3,), (2,)])])

# env.step() returns a tuple containing one element, the observation (a TimeStep object)
# timestep = env.step([actions.FunctionCall(1, [(20,20)])])
# obs = timestep[0]


def select_each_and_move(env):
   """
   This will select each unit iteratively
   And each time, give it an order to move at a given point
   To do so, we select all units, and then select one of them using unit_id
   """

   # TODO: is the unit_id consistent though ? probably not...
   # TODO: need to find another way to select each unit in a loop
   # One solution would be to create control groups, but we only have 10 control groups possible
   # And we might end up with more than the initial 9 marines...
   # So we'd need to create control groups of multiple units after the 10th marine

   for i in range(9):
      # Select all units: just select a rectangle covering all the screen
      # Since in this minigame we have a full view (no need to deal with camera)
      print('selecting all units')
      obs = env.step([actions.FunctionCall(_SELECT_RECT, [(False,), (0, 0), (83, 83)])])[0]

      time.sleep(1)

      # Now select one of these units:
      if _SELECT_UNIT not in obs.observation["available_actions"]:
         print('WARNING: select unit not in available actions')
         break

      else:
         # Select ith unit
         print('selecting {}th unit'.format(i))
         obs = env.step([actions.FunctionCall(_SELECT_UNIT, [(0,), (i,)])])[0]
         time.sleep(1)

      # Now tell this unit to attack a point on the screen:
      if _MOVE_SCREEN not in obs.observation["available_actions"]:
         print('WARNING: attack screen not in available actions')
         break

      else:
         # Move to point point (83, 10*i), e.g. unit 3 attacks point (83, 30)
         print('giving move order to {}th unit'.format(i))
         # obs = env.step([actions.FunctionCall(_ATTACK_SCREEN, [(False,), (10*i, 10*i)])])[0]
         obs = env.step([actions.FunctionCall(_MOVE_SCREEN, [(False,), (83, 10*i)])])[0]
         time.sleep(1)


   time.sleep(5) # wait for all units to be placed
   return

def select_one_and_move(env):
   """
   This function lets us verify consistency in the action 'select_unit'
   i.e. are the unit_ids the same when you select a rectangle, select one of the units by its id
   and repeat this multiple times ? This is a critical concern for our project
   RESULT: it looks that there is some kind of consistency, but we need to test in more tricky cases
           also, what happens when id number i ? is the unit_id reassigned ?
   """
   for i in range(20):
      print('selecting all units')
      obs = env.step([actions.FunctionCall(_SELECT_RECT, [(False,), (0, 0), (83, 83)])])[0]
      time.sleep(1)

      # Select unit 4:
      if _SELECT_UNIT not in obs.observation["available_actions"]:
         print('WARNING: select unit not in available actions')
         break
      else:
         print('selecting 4th unit')
         obs = env.step([actions.FunctionCall(_SELECT_UNIT, [(0,), (4,)])])[0]
         time.sleep(1)

      # Now tell this unit to attack a point on the screen:
      if _MOVE_SCREEN not in obs.observation["available_actions"]:
         print('WARNING: attack screen not in available actions')
         break

      else:
         print('giving move order to {}th unit'.format(i))
         obs = env.step([actions.FunctionCall(_MOVE_SCREEN, [(False,), (83, 4*i)])])[0]
         time.sleep(1)

select_each_and_move(env)
# select_one_and_move(env)

print('Done, closing environment...')
env.close() # this will close the window game
