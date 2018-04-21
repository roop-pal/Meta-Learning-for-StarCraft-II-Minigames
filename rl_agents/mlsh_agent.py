from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import math
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

# from agents.mlsh_network import build_net
from rl_agents.mlsh_network import build_net
import utils as U


class MLSHAgent(object):
  """An agent specifically for solving the mini-game maps."""
  def __init__(self, training, msize, ssize, num_subpol, subpol_steps, num_thread, name='MLSH/MLSHAgent'):

    print("NAME", name)

    self.name = name
    self.training = training
    self.summary, self.subpol_summary = [], []
    # Minimap size, screen size and info size
    assert msize == ssize
    self.msize = msize
    self.ssize = ssize
    self.isize = len(actions.FUNCTIONS)
    self.num_subpol = num_subpol
    self.subpol_steps = subpol_steps
    self.num_thread = num_thread

    self.ep_subpol_choices = []
    self.train_only_master = True
    self.test_run = False

    self.steps_on_subpol = 0
    self.cur_subpol = 0

    self.count_steps = 0
    self.test_scores = []

  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    # Epsilon schedule
    self.epsilon = [0.05, 0.2]


  def reset_master(self):

    print("GLOBAL VARIABLES FOR THREAD " + str(self.num_thread) + ":")

    variables_names = [v.name for v in tf.global_variables()]
    values = self.sess.run(variables_names)
    for k, v in zip(variables_names, values):
      print("Variable: ", k)
      print("Shape: ", v.shape)

    master_vars = []

    for v in tf.trainable_variables():
      if v.name.startswith('subpol_choice_' + str(self.num_thread)) \
              or v.name.startswith('master_value_' + str(self.num_thread)):
        master_vars.append(v)

    print(master_vars)

    self.sess.run(tf.variables_initializer(master_vars))


  def build_subpolicy(self, opt, pol_id, reuse):

    with tf.variable_scope('subpol'):

      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      self.spatial_action = self.spatial_actions[pol_id]
      self.non_spatial_action = self.non_spatial_actions[pol_id]

      spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
      spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
      non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
      valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
      valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
      self.subpol_summary.append(tf.summary.histogram('spatial_action_prob_' + str(pol_id), spatial_action_prob))
      self.subpol_summary.append(tf.summary.histogram('non_spatial_action_prob_' + str(pol_id), non_spatial_action_prob))

      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
      advantage = tf.stop_gradient(self.value_target - self.value)
      policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      value_loss = - tf.reduce_mean(self.value * advantage)
      self.subpol_summary.append(tf.summary.scalar('policy_loss_' + str(pol_id), policy_loss))
      self.subpol_summary.append(tf.summary.scalar('value_loss_' + str(pol_id), value_loss))

      # TODO: policy penalty
      loss = policy_loss + value_loss

      grads = opt.compute_gradients(loss)

      cliped_grad = []
      for grad, var in grads:
        # Ignore gradients for other output layers for other subpolicies
        if grad == None:
          continue
        self.subpol_summary.append(tf.summary.histogram(var.op.name, var))
        self.subpol_summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)
        cliped_grad.append([grad, var])

      self.subpol_train_ops.append(opt.apply_gradients(cliped_grad))
      # self.summary_op = tf.summary.merge(self.summary)

  def build_master_policy(self, opt):
    # Create targets and masks:
    self.subpolicy_selected = tf.placeholder(tf.float32, [None, self.num_subpol], name='subpolicy_selected')

    advantage = tf.stop_gradient(self.value_target - self.master_value)
    # value_target is here master_value_target actually
    subpol_choice_prob = tf.reduce_sum(self.subpol_choice * self.subpolicy_selected, axis=1)
    subpol_choice_log_prob = tf.log(tf.clip_by_value(subpol_choice_prob, 1e-10, 1.))
    master_policy_loss = - tf.reduce_mean(subpol_choice_log_prob * advantage)
    master_value_loss = - tf.reduce_mean(self.master_value * advantage)

    self.summary.append(tf.summary.scalar('master_policy_loss', master_policy_loss))
    self.summary.append(tf.summary.scalar('master_value_loss', master_value_loss))

    self.summary.append(tf.summary.histogram('subpol_choice', self.subpol_choice))

    master_loss = master_policy_loss + master_value_loss

    grads = opt.compute_gradients(master_loss)

    cliped_grad = []
    for grad, var in grads:
      # assert grad != None

        # TODO: MAKE THIS WORK WITH MULTIPLE THREADS

      if grad is None or ('master_value' not in var.name and 'subpol_choice' not in var.name):
        # TODO: find a better way to stop gradients in first layers
        # compute_gradients computes grads for some variables not needed / related
        continue
      self.summary.append(tf.summary.histogram(var.op.name, var))
      self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
      grad = tf.clip_by_norm(grad, 10.0)
      cliped_grad.append([grad, var])

      self.master_train_op = opt.apply_gradients(cliped_grad)

  def build_model(self, reuse, dev, ntype):

    print("NAME:", self.name)

    with tf.variable_scope(self.name) and tf.device(dev):

      print("build_model scope: " + str(tf.get_variable_scope().name))

      # Set inputs of networks
      self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype, self.num_subpol, reuse, self.num_thread)
      self.spatial_actions, self.non_spatial_actions, self.value, self.master_value, self.subpol_choice = net

      # Create training operation for the subpolicies:
      # Set targets and masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      self.subpol_train_ops = []

      with tf.variable_scope('train_vars'):

        # Build the optimizer
        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
        opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)

        for pol_id in range(self.num_subpol):
          self.build_subpolicy(opt, pol_id, reuse)

        # Create training operation for the master policy:
        self.build_master_policy(opt)

        self.summary_op = tf.summary.merge(self.summary)
        self.subpol_summary_op = tf.summary.merge(self.subpol_summary)

        self.score = tf.placeholder(tf.float32, name='episode_score')
        self.score_summary_op = tf.summary.scalar('episode_score', self.score)

        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)


  def choose_subpolicy(self, minimap, screen, info):
      # Get softmax outputs for choosing subpolicy
      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      subpol_choice = self.sess.run(
        self.subpol_choice,
        feed_dict=feed)

      # print('SUBPOLICY CHOICE: ' + str(subpol_choice))

      # Choose max probability output for subpolicy
      subpol_choice = np.argmax(subpol_choice)

      # Choose subpolicy using epsilon-greedy method
      if self.training and np.random.rand() < self.epsilon[1]:
        # self.cur_subpol = np.random.randint(0, self.num_subpol)
        subpol_choice = np.random.randint(0, self.num_subpol)

      return subpol_choice

  def choose_action(self, minimap, screen, info, obs):
    # Run the graph for the current subpolicy to get action
    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    non_spatial_action, spatial_action = self.sess.run(
      [self.non_spatial_actions[self.cur_subpol], self.spatial_actions[self.cur_subpol]],
      feed_dict=feed)

    # Select an action and a spatial target
    non_spatial_action = non_spatial_action.ravel()
    spatial_action = spatial_action.ravel()
    valid_actions = obs.observation['available_actions']
    act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
    target = np.argmax(spatial_action)
    target = [int(target // self.ssize), int(target % self.ssize)]

    # Epsilon greedy exploration
    if self.training and np.random.rand() < self.epsilon[0]:
      act_id = np.random.choice(valid_actions)

    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-4, 5)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-4, 5)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    # Set act_id and act_args
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])  # TODO: Be careful
    
    return act_id, act_args

  def step(self, obs):
    """
    Given observation obs:
    - choose subpolicy to follow (if it's the right time to change)
    - choose action to perform according to this policy
    """
    minimap, screen, info = U.preprocess_obs(obs, self.isize)

    # Only change master's choice of subpolicy every self.subpol_steps steps:
    if self.steps_on_subpol % self.subpol_steps == 0:
      self.cur_subpol = self.choose_subpolicy(minimap, screen, info)
      self.steps_on_subpol = 0

    # Store subpol_choice at each step for later call to update()
    self.ep_subpol_choices.append(self.cur_subpol)

    act_id, act_args = self.choose_action(minimap, screen, info, obs)

    return actions.FunctionCall(act_id, act_args)

  def update_subpolicies(self, rbs, disc, lr, cter, minimaps, screens, infos):
    """
    Given (already reversed) replay buffer:
    - computes value for each step
    - update the subpolicy networks and the first common layers (state representation)
    """
    # Compute R, which is value of the last observation
    obs = rbs[0][-1]
    if obs.last():
      R = 0
    else:
      minimap, screen, info = U.preprocess_obs(obs, self.isize)
      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value, feed_dict=feed)

    # print('R subpolicy: ', R)

    value_target = np.zeros([len(rbs)], dtype=np.float32)
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32)
    valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

    for i, [obs, action, next_obs] in enumerate(rbs):
      reward = obs.reward
      act_id = action.function
      act_args = action.arguments

      value_target[i] = reward + disc * value_target[i-1]

      valid_actions = obs.observation["available_actions"]
      valid_non_spatial_action[i, valid_actions] = 1
      non_spatial_action_selected[i, act_id] = 1

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    # Update each subpolicy using gradient descent on the steps of this episode for which that
    # subpolicy was being used
    for pol_id in range(self.num_subpol):
      pol_inds = np.where(np.array(self.ep_subpol_choices) == pol_id)[0]

      # No game steps in this episode used this policy
      if len(pol_inds) == 0:
        continue

      # Train
      feed = {self.minimap: minimaps[pol_inds],
              self.screen: screens[pol_inds],
              self.info: infos[pol_inds],
              self.value_target: value_target[pol_inds],
              self.valid_spatial_action: valid_spatial_action[pol_inds],
              self.spatial_action_selected: spatial_action_selected[pol_inds],
              self.valid_non_spatial_action: valid_non_spatial_action[pol_inds],
              self.non_spatial_action_selected: non_spatial_action_selected[pol_inds],
              self.learning_rate: lr}
      _, summary = self.sess.run([self.subpol_train_ops[pol_id], self.subpol_summary_op], feed_dict=feed)
      self.summary_writer.add_summary(summary, cter)

  def update_master_policy(self, rbs, master_disc, lr, cter, minimaps, screens, infos):
    """
    TODO: master policy update shouldn't affect the layers before feat_fc i.e. the input transformation
    """
    obs = rbs[0][-1]
    if obs.last():
      R_master = 0
    else:
      minimap, screen, info = U.preprocess_obs(obs, self.isize)
      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R_master = self.sess.run(self.master_value, feed_dict=feed)
    
    score_summary = self.sess.run(self.score_summary_op, feed_dict={self.score: obs.observation['score_cumulative'][0]})
    self.summary_writer.add_summary(score_summary, cter)

    # print('R_master: ', R_master)

    # note there is something to figure out with learning rate
    n_master_steps = math.ceil(len(rbs) / self.subpol_steps)

    # get decisions made by master policy every self.subpol_steps steps:
    subpolicy_selected = np.zeros([n_master_steps, self.num_subpol])
    for i, policy_index in enumerate(self.ep_subpol_choices):
      if i % self.subpol_steps == 0:
        subpolicy_selected[int(i/self.subpol_steps), policy_index] = 1

    # sum rewards gotten between each change of subpolicy and compute values:
    master_value_target = np.zeros([n_master_steps], dtype=np.float32)
    master_value_target[-1] = R_master

    for i in range(n_master_steps):
      sum_rewards = sum([obs.reward for obs,_,_ in rbs[i:(i+self.subpol_steps)]])
      master_value_target[i] = sum_rewards + master_disc * master_value_target[i-1]

    master_choice_inds = list(range(0, len(rbs), self.subpol_steps))

    feed = {self.minimap: minimaps[master_choice_inds],
            self.screen: screens[master_choice_inds],
            self.info: infos[master_choice_inds],
            self.value_target: master_value_target,
            self.learning_rate: lr,
            self.subpolicy_selected: subpolicy_selected}
    _, summary = self.sess.run([self.master_train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)

  def update(self, rbs, disc, lr, cter):

    # If doing a test run, get the cumulative score over full minigame (rather than just max_agent_steps steps)
    obs = rbs[-1][-1]
    # Print out score on a test run through a full episode, don't update network on test run
    if self.test_run and obs.last():
      self.test_scores.append(obs.observation['score_cumulative'][0])
      self.ep_subpol_choices = []
      return

    print("Total game steps: " + str(self.count_steps))

    # reverse the replay buffer in order to calculate values:
    rbs.reverse()
    self.ep_subpol_choices.reverse()

    # process the observations from the replay to use them for the update:
    minimaps, screens, infos = U.preprocess_rbs(rbs, self.isize)

    master_disc = disc # TODO: tune this
    self.update_master_policy(rbs, master_disc, lr, cter, minimaps, screens, infos)

    # train only the master policy, increment steps by number of master policy choices observed
    if self.train_only_master:
      self.count_steps += int(len(rbs) / self.subpol_steps)

    else:
      self.count_steps += len(rbs)
      self.update_subpolicies(rbs, disc, lr, cter, minimaps, screens, infos)

    self.ep_subpol_choices = []

  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
