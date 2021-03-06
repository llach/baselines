import time
import json
import functools
import datetime
import numpy as np
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util, colorize
from baselines.common.policies import build_policy

from forkan.common.utils import log_alg
from forkan.common.csv_logger import CSVLogger
from forkan.common.tf_utils import vector_summary, scalar_summary

from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner
from baselines.common.tf_util import get_session

import datetime

from tensorflow import losses

from tqdm import tqdm

import matplotlib.pyplot as plt


class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = self.step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e7),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    reward_average=20,
    log_interval=100,
    load_path=None,
    env_id=None,
    play=False,
    save=True,
    tensorboard=False,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''



    set_global_seeds(seed)

    if hasattr(env, 'vae_name'):
        vae = env.vae_name.split('lat')[0][:-1]
    else:
        vae = None

    savepath, env_id_lower = log_alg('a2c-debug', env_id, locals(), vae, num_envs=env.num_envs, save=save, lr=lr)

    csv_header = ['timestamp', "nupdates", "total_timesteps", "fps", "policy_entropy", "value_loss",
                  "explained_variance", "mean_reward [{}]".format(reward_average), "nepisodes"]
    csv = CSVLogger('{}progress.csv'.format(savepath), *csv_header)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Calculate the batch_size
    nbatch = nenvs * nsteps

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon,
                  total_timesteps=total_timesteps, lrschedule=lrschedule)

    if load_path is not None:
        print('loading model ... ')
        model.load(load_path)

        if play:
            return model

    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    # Start total timer
    tstart = time.time()

    episode_rewards = []
    current_rewards = [0.0] * nenvs
    nepisodes = 0

    best_rew = -np.infty

    if tensorboard:
        print('logging to tensorboard')
        s = get_session()
        import os
        fw = tf.summary.FileWriter('{}/a2c/{}/'.format(os.environ['HOME'], savepath.split('/')[-2]), s.graph)

        ft = None
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in vars:
            if v.name == 'enc-conv-3/kernel:0': ft = v

        def ftv_out():
            if ft is not None:
                ftv = np.mean(s.run([ft]), axis=-1)
                ftv = np.mean(ftv)
                print(ftv)
                return ftv
            else:
                0

        pl_ph = tf.placeholder(tf.float32, (), name='policy-loss')
        pe_ph = tf.placeholder(tf.float32, (), name='policy-entropy')
        vl_ph = tf.placeholder(tf.float32, (), name='value-loss')
        rew_ph = tf.placeholder(tf.float32, (), name='reward')
        ac_ph = tf.placeholder(tf.float32, (nbatch, 1), name='actions')
        ac_clip_ph = tf.placeholder(tf.float32, (nbatch, 1), name='actions')

        weight_ph = tf.placeholder(tf.float32, (), name='encoder-kernel')
        scalar_summary('encoder-conv-kernel', weight_ph)

        tf.summary.histogram('actions-hist', ac_ph)
        tf.summary.histogram('actions-hist-clipped', ac_clip_ph)

        scalar_summary('reward', rew_ph)

        scalar_summary('value-loss', vl_ph)
        scalar_summary('policy-loss', pl_ph)
        scalar_summary('policy-entropy', pe_ph)
        vector_summary('actions', ac_ph)

        merged_ = tf.summary.merge_all()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, dones, raw_rewards = runner.run()

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        for n, (rs, ds) in enumerate(zip(raw_rewards, dones)):
            rs = rs.tolist()
            ds = ds.tolist()

            for r, d in zip(rs, ds):
                if d:
                    episode_rewards.append(current_rewards[n])
                    current_rewards[n] = 0.0
                else:
                    current_rewards[n] += r

        if len(episode_rewards) > reward_average:
            mrew = np.mean(episode_rewards[-reward_average:])
        else:
            mrew = -np.infty

        if np.any(dones):
            nepisodes += 1

        if mrew > best_rew:
            logger.log('model improved from {} to {}. saving ...'.format(best_rew, mrew))
            model.save('{}weights_best'.format(savepath))
            best_rew = mrew

        # Calculates if value function is a good predicator of the returns (ev > 1)
        # or if it's just worse than predicting nothing (ev =< 0)
        ev = explained_variance(values, rewards)

        # Calculate the fps (frame per second)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)

        csv.writeline(datetime.datetime.now().isoformat(), update, update*nbatch, fps, float(policy_entropy), float(value_loss), float(ev),
                      float(mrew), nepisodes)

        if tensorboard:
            summary = s.run(merged_, feed_dict={
                pl_ph: policy_loss,
                pe_ph: policy_entropy,
                vl_ph: value_loss,
                rew_ph: mrew,
                ac_ph: actions,
                ac_clip_ph: np.clip(actions, -2, 2),
                weight_ph: ftv_out(),
            })

            fw.add_summary(summary, update)

        if update % log_interval == 0 or update == 1:
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("mean_reward [{}]".format(reward_average), float(mrew))
            logger.record_tabular("nepisodes", nepisodes)
            logger.dump_tabular()

            perc = ((update * nbatch) / total_timesteps) * 100
            steps2go = total_timesteps - (update * nbatch)
            secs2go = steps2go / fps
            min2go = secs2go / 60

            hrs = int(min2go // 60)
            mins = int(min2go) % 60
            print(colorize('ETA: {}h {}min | done {}% '.format(hrs, mins, int(perc)), color='cyan'))

    model.save('{}weights_latest'.format(savepath))

    return model

