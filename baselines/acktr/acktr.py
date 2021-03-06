import os.path as osp
import time
import functools
import datetime
import numpy as np
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, save_variables, load_variables

from baselines.a2c.runner import Runner
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.acktr import kfac

from tqdm import tqdm

from forkan.common.utils import log_alg
from forkan.common.csv_logger import CSVLogger

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs,total_timesteps, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', is_async=True):

        self.sess = sess = get_session()
        nbatch = nenvs * nsteps
        with tf.variable_scope('acktr_model', reuse=tf.AUTO_REUSE):
            self.model = step_model = policy(nenvs, 1, sess=sess)
            self.model2 = train_model = policy(nenvs*nsteps, nsteps, sess=sess)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        self.logits = train_model.pi

        ##training loss
        pg_loss = tf.reduce_mean(ADV*neglogpac)
        entropy = tf.reduce_mean(train_model.pd.entropy())
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.losses.mean_squared_error(tf.squeeze(train_model.vf), R)
        train_loss = pg_loss + vf_coef * vf_loss


        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neglogpac)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params=params = find_trainable_variables("acktr_model")

        self.grads_check = grads = tf.gradients(train_loss,params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip,\
                momentum=0.9, kfac_update=1, epsilon=0.01,\
                stats_decay=0.99, is_async=is_async, cold_iter=10, max_grad_norm=max_grad_norm)

            # update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, PG_LR:cur_lr, VF_LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

def learn(network, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
                 ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear', load_path=None, is_async=True,
          reward_average=20, vae='', env_id=None, play=False, save=True, **network_kwargs):
    set_global_seeds(seed)

    if network == 'cnn':
        network_kwargs['one_dim_bias'] = True

    savepath, env_id_lower = log_alg('acktr', env_id, locals(), vae, num_envs=env.num_envs, save=save)
    csv_header = ['timestamp', "nupdates", "total_timesteps", "fps", "policy_entropy", "value_loss",
                  "explained_variance", "mean_reward [{}]".format(reward_average), "nepisodes"]
    csv = CSVLogger('{}progress.csv'.format(savepath), *csv_header)

    policy = build_policy(env, network, **network_kwargs)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda : Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
                                =nsteps, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule, is_async=is_async)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    if load_path is not None:
        model.load(load_path)

    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    if is_async:
        enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)
    else:
        enqueue_threads = []

    episode_rewards = []
    current_rewards = [0.0]*nenvs
    nepisodes = 0

    for update in tqdm(range(1, total_timesteps//nbatch+1)):
        obs, states, rewards, masks, actions, values, dones, raw_rewards = runner.run()

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

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)

        try:
            ev = explained_variance(values, rewards)
        except:
            print("caught ev calc error")
            ev = -5

        csv.writeline(datetime.datetime.now().isoformat(), update, update * nbatch, fps, float(policy_entropy),
                      float(value_loss), float(ev),
                      float(mrew), nepisodes)

        if update % log_interval == 0 or update == 1:

            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("mean_reward [{}]".format(reward_average), float(mrew))
            logger.record_tabular("nepisodes", nepisodes)
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    return model
