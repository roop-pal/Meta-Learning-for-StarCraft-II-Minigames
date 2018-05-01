from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

import logging

logger = logging.getLogger('starcraft_agent')

def build_net(minimap, screen, info, msize, ssize, num_action, ntype, num_subpol, reuse, num_thread):
  if ntype == 'atari':
    return build_atari(minimap, screen, info, msize, ssize, num_action, num_subpol, reuse, num_thread)
  elif ntype == 'fcn':
    return build_fcn(minimap, screen, info, msize, ssize, num_action)
  else:
    raise 'FLAGS.net must be atari or fcn'


def build_atari(minimap, screen, info, msize, ssize, num_action, num_subpol, reuse, num_thread):
  """
  Builds subpolicies and master policy which share an atari-net initial set of feature layers
  """

  # Create a variable scope for variables which are reused across different threads (i.e. the subpolicy parameters)
  with tf.variable_scope("reuse_net"):

      if reuse:
          tf.get_variable_scope().reuse_variables()
          assert tf.get_variable_scope().reuse

      mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                             num_outputs=16,
                             kernel_size=8,
                             stride=4,
                             scope='mconv1', reuse=reuse)
      mconv2 = layers.conv2d(mconv1,
                             num_outputs=32,
                             kernel_size=4,
                             stride=2,
                             scope='mconv2')
      sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                             num_outputs=16,
                             kernel_size=8,
                             stride=4,
                             scope='sconv1')
      sconv2 = layers.conv2d(sconv1,
                             num_outputs=32,
                             kernel_size=4,
                             stride=2,
                             scope='sconv2')
      info_fc = layers.fully_connected(layers.flatten(info),
                                       num_outputs=256,
                                       activation_fn=tf.tanh,
                                       scope='info_fc')

      # Compute spatial actions, non spatial actions and value
      feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
      feat_fc = layers.fully_connected(feat_fc,
                                       num_outputs=256,
                                       activation_fn=tf.nn.relu,
                                       scope='feat_fc')


      spatial_actions = []
      non_spatial_actions = []

      # Create separate action output layers for each subpolicy
      for pol_i in range(1, num_subpol + 1):

          # Determine action (i.e. Select, Attack, Hotkey 1, etc.)
          non_spatial_action = layers.fully_connected(feat_fc,
                                                      num_outputs=num_action,
                                                      activation_fn=tf.nn.softmax,
                                                      scope=('non_spatial_action_' + str(pol_i)))

          # Determine location at which to perform the chosen action (if it is a action which acts at a location)
          spatial_action_x = layers.fully_connected(feat_fc,
                                                    num_outputs=ssize,
                                                    activation_fn=tf.nn.softmax,
                                                    scope=('spatial_action_x_' + str(pol_i)))
          spatial_action_y = layers.fully_connected(feat_fc,
                                                    num_outputs=ssize,
                                                    activation_fn=tf.nn.softmax,
                                                    scope=('spatial_action_y_' + str(pol_i)))
          spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
          spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
          spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
          spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
          spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

          non_spatial_actions.append(non_spatial_action)
          spatial_actions.append(spatial_action)

      # Value function shared by all subpolicies
      value = tf.reshape(layers.fully_connected(feat_fc,
                                                num_outputs=1,
                                                activation_fn=None,
                                                scope='value'), [-1])

  # Note: master policy parameters below are outside of variable scope "reuse_net" since they are not reused across threads
  # Value function for the master policy
  master_value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='master_value_'+str(num_thread)), [-1])

  # Choose a subpolicy to use
  subpol_choice = layers.fully_connected(feat_fc,
                                        num_outputs=num_subpol,
                                        activation_fn=tf.nn.softmax,
                                        scope='subpol_choice_'+str(num_thread))

  # Get the variables corresponding to the master policy layers above, used for resetting master policy
  logger.debug('Master variables:')
  master_vars = []
  for var in tf.trainable_variables():
      if 'master_value_'+str(num_thread) in var.name or 'subpol_choice_'+str(num_thread) in var.name:
        master_vars.append(var)
        logger.debug('Variable: %s', var.name)

  return spatial_actions, non_spatial_actions, value, master_value, subpol_choice, master_vars


def build_fcn(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions
  feat_conv = tf.concat([mconv2, sconv2], axis=3)
  spatial_action = layers.conv2d(feat_conv,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')
  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value
