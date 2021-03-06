import datetime
import time
from collections import deque

import numpy as np
from forkan.common.csv_logger import CSVLogger
from forkan.common.tf_utils import scalar_summary
from forkan.common.utils import log_alg

import baselines.common.tf_util as U
from baselines import logger
from baselines.common import explained_variance, set_global_seeds, colorize
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner


def constfn(val):
    def f(_):
        return val
    return f

def get_theta_traversal_tensor(latents=5, frames=200):
    import cairocffi as cairo
    import tensorflow as tf
    import tfmpl

    @tfmpl.figure_tensor
    def theta_traversal(mus, xs):
        ''' Draw theta traversal plots. '''
        figs = tfmpl.create_figures(1, figsize=(12, 10))
        ax = figs[0].add_subplot(111)
        for i in range(mus.shape[0]):
            ax.plot(xs, mus[i], label=f'mus[{i}]')
            ax.scatter(xs, mus[i], linewidths=0.05, marker='x')
        figs[0].tight_layout()

        return figs

    mus_ph = tf.placeholder(tf.float32, (latents, frames))

    image_tensor = theta_traversal(mus_ph, xs=np.linspace(0, 2 * np.pi, frames))
    image_summary = tf.summary.image('theta-traversal', image_tensor)

    w, h = 64, 64
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)

    def _render_pendulum(th):
        cr = cairo.Context(surf)

        # draw background
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        # apply transforms
        cr.translate((w / 2), h / 2)
        cr.rotate(np.pi - th)

        # draw shapes that form the capsule
        cr.rectangle(-2.5, 0, 5, 27)
        cr.arc(0, 0, 2.5, 0, 2 * np.pi)
        cr.arc(0, (h / 2) - 4, 2.5, 0, 2 * np.pi)

        # draw color
        cr.set_source_rgb(.8, .3, .3)
        cr.fill()

        # center sphere
        cr.arc(0, 0, 1, 0, 2 * np.pi)
        cr.set_source_rgb(0, 0, 0)
        cr.fill()

        # reshape, delete fourth (alpha) channel, greyscale and normalise
        return np.expand_dims(np.dot(np.frombuffer(surf.get_data(), np.uint8).reshape([w, h, 4])[..., :3],
                                     [0.299, 0.587, 0.114]), -1) / 255

    frames_arr = np.zeros([frames, 64, 64, 1])

    for i, theta in enumerate(np.linspace(0, 2 * np.pi, frames)):
        frames_arr[i] = _render_pendulum(theta)

    return mus_ph, image_summary, frames_arr

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4, target_kl=0.01,
          vf_coef=0.5, pg_coef=1.0, max_grad_norm=0.5, gamma=0.99, lam=0.95, rl_coef=1.0, v_net='pendulum', f16=False,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2, vae_params=None, log_weights=False, early_stop=False,
          save_interval=50, load_path=None, model_fn=None, env_id=None, play=False, save=True, tensorboard=False, k=None,
          alpha=1.0, plot_thetas=False, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    if hasattr(env, 'vae_name'):
        vae = env.vae_name.split('lat')[0][:-1]
    else:
        vae = None

    if vae_params is None:
        models = 'd'
        with_vae = False
        with_kl = False
        rl_coef = 1
        scaled_v = True
    else:
        with_vae = True

        if 'scaled_re_loss' in vae_params:
            scaled_v = vae_params['scaled_re_loss']
        else:
            scaled_v = True

        if 'init_from' in vae_params:
            models = 'retrain-' + vae_params['init_from'].split('lat')[0][:-1].split('-')[-1]
        else:
            if 'beta' in vae_params:
                models = 'scratch-b' + str(vae_params['beta'])
            else:
                models = 'scratch-b1.0'

        if 'with_kl' in vae_params:
            with_kl = True
            vae_params.pop('with_kl')
        else:
            with_kl = False

    if 'latents' in network_kwargs:
        latents = network_kwargs['latents']
    else:
        latents = None

    savepath, env_id_lower = log_alg('ppo2', env_id, locals(), vae, num_envs=env.num_envs, save=save, lr=lr, k=k,
                                     seed=seed, model=models, with_kl=with_kl, rl_coef=rl_coef, early_stop=early_stop,
                                     target_kl=target_kl, scaled_re_loss=scaled_v, alpha=alpha, latents=latents)

    # Instantiate the model object (that creates act_model and train_model)
    if vae_params is None:
        from baselines.ppo2.model import Model
        model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                      nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm)
        if load_path is not None:
            model.load(load_path)
            print(model)
            if play:
                return model
        # Instantiate the runner object
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
        if eval_env is not None:
            eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)
    else:
        from baselines.ppo2.vae_model import VAEModel
        from baselines.ppo2.vae_runner import VAERunner

        model = VAEModel(k=k, policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                         nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, pg_coef=pg_coef, savepath=load_path or savepath, env=env,vae_params=vae_params,
                         max_grad_norm=max_grad_norm, with_kl=with_kl, rl_coef=rl_coef, v_net=v_net, alpha=alpha)
        if load_path is not None:
            model.load(load_path)
            if play:
                return model
        # Instantiate the runner object
        runner = VAERunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

        if eval_env is not None:
            eval_runner = VAERunner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    csv_header = ['timestamp', "nupdates", "total_timesteps", "fps", "policy_entropy", "value_loss", "policy_loss",
                  "explained_variance", "mean_reward", "approx_kl", "clip_frac", "stop"]

    if with_vae:
        csv_header +=['rec-loss', 'kl-loss'] + ['z{}-kl'.format(i) for i in range(model.vae.latent_dim)]

    csv = CSVLogger('{}progress.csv'.format(savepath), *csv_header)

    if f16:

        import  tensorflow as tf

        print('casting to bfloat16')
        for var in tf.global_variables():
            if (var.dtype == np.float32):
                tf.add_to_collection('assignOps', var.assign(
                    tf.cast(var, tf.bfloat16)))
            else:
                tf.add_to_collection('assignOps', var.assign(var))
        get_session().run(tf.get_collection('assignOps'))

    import tensorflow as tf
    if with_vae:
        vf_grad_op = tf.gradients(model.train_model.vf, model.vae.X)

    buf = []
    t = 0
    if tensorboard:
        print('logging to tensorboard')
        s = get_session()
        import os
        fw = tf.summary.FileWriter('{}/ppo2/{}/'.format(os.environ['HOME'], savepath.split('/')[-2]), s.graph)

        with tf.variable_scope('losses'):
            pl_ph = tf.placeholder(tf.bfloat16, (), name='policy-loss')
            pe_ph = tf.placeholder(tf.bfloat16, (), name='policy-entropy')
            vl_ph = tf.placeholder(tf.bfloat16, (), name='value-loss')

        with tf.variable_scope('stopping'):
            ak_ph = tf.placeholder(tf.bfloat16, (), name='approx-kl')
            cf_ph = tf.placeholder(tf.bfloat16, (), name='clipfrac')
            stop_ph = tf.placeholder(tf.uint8, (), name='stopped')

        rew_ph = tf.placeholder(tf.bfloat16, (), name='reward')
        ac_ph = tf.placeholder(tf.bfloat16, (nbatch, 1), name='actions')
        ac_clip_ph = tf.placeholder(tf.bfloat16, (nbatch, 1), name='actions-clipped')

        sums = []

        sums.append(scalar_summary('approx-kl', ak_ph, scope='stopping'))
        sums.append(scalar_summary('clipfrac', cf_ph, scope='stopping'))
        sums.append(scalar_summary('stopped', stop_ph, scope='stopping'))

        sums.append(tf.summary.histogram('actions-hist', ac_ph))
        sums.append(tf.summary.histogram('actions-hist-clipped', ac_clip_ph))

        sums.append(scalar_summary('reward', rew_ph))

        sums.append(scalar_summary('value-loss', vl_ph, scope='rl-loss'))
        sums.append(scalar_summary('policy-loss', pl_ph, scope='rl-loss'))
        sums.append(scalar_summary('policy-entropy', pe_ph, scope='rl-loss'))

        if log_weights:
            if with_vae:
                vvs = tf.trainable_variables('vae')
                pvs = tf.trainable_variables('ppo2_model')

                with tf.variable_scope('policy-weights'):
                    for v in pvs:
                        sums.append(tf.summary.histogram('{}'.format(v.name), v))

                with tf.variable_scope('vae-weights'):
                    for v in vvs:
                        sums.append(tf.summary.histogram('{}'.format(v.name), v))

            else:
                pvs = tf.trainable_variables('ppo2_model')

                with tf.variable_scope('policy-weights'):
                    for v in pvs:
                        sums.append(tf.summary.histogram('{}'.format(v.name), v))

        if with_vae:
            rel_ph = tf.placeholder(tf.bfloat16, (), name='rec-loss')
            kll_ph = tf.placeholder(tf.bfloat16, (), name='kl-loss')
            klls_ph = [tf.placeholder(tf.bfloat16, (), name=f'z{i}-kl') for i in range(model.vae.latent_dim)]

            sums.append(scalar_summary('reconstruction-loss', rel_ph, scope='vae-loss'))
            sums.append(scalar_summary('kl-loss', kll_ph, scope='vae-loss'))
            for i in range(model.vae.latent_dim):
                sums.append(scalar_summary(f'z{i}-kl', klls_ph[i], scope='z-kl'))

        merged_ = tf.summary.merge(sums)

        if plot_thetas:
            img_ph = tf.placeholder(tf.float32, shape=(1,) + tuple(np.multiply(ob_space.shape[:-1], [5, 3])) + (3,))
            im_sum = tf.summary.image('images', img_ph, max_outputs=5)

            mus_ph, th_im_sum, theta_frames = get_theta_traversal_tensor()
            theta_frames = theta_frames.reshape([40, 5, 64, 64, 1])

    var_list = tf.trainable_variables()

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)

    best_rew = -np.inf
    print('starting main loop ...')
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        """ This is for observation debugging. Uncomment with caution. """
        # import matplotlib.pyplot as plt
        # print(obs.shape)
        # for i in range(nsteps):
        #     buf.append(obs[i])
        #
        # for idx in range(obs.shape[-1]):
        #     plt.plot(np.asarray(buf)[:, idx])
        # plt.show()
        # if t == 50: exit(0)
        # t += 1
        
        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        re_l = []
        kl_l = []
        kl_ls = []

        thold = get_flat()

        if states is None or states == []: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    if with_vae:
                        if _ == (noptepochs - 1):
                            fw_for_model = fw
                        else:
                            fw_for_model = None
                        res, kl_losses, re_loss, kl_loss = model.train(lrnow, cliprangenow, *slices,
                                                                       fw=fw_for_model, sumstep=update*nbatch)
                        mblossvals.append(res)
                        re_l.append(re_loss)
                        kl_l.append(kl_loss)
                        kl_ls.append(kl_losses)
                    else:
                        res = model.train(lrnow, cliprangenow, *slices)
                        mblossvals.append(res)
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    if with_vae:

                        res, kl_losses, re_loss, kl_loss = model.train(lrnow, cliprangenow, *slices, mbstates)
                        mblossvals.append(res)
                        re_l.append(re_loss)
                        kl_l.append(kl_loss)
                        kl_ls.append(kl_losses)
                    else:
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        re_l = np.mean(re_l, axis=0)
        kl_l = np.mean(kl_l, axis=0)
        kl_ls = np.mean(kl_ls, axis=0)
        mrew = safemean([epinfo['r'] for epinfo in epinfobuf])

        stop = False
        if early_stop and (lossvals[-2] > 1.5 * target_kl):
            logger.log(f'KL {lossvals[-2]} violates constraint {1.5 * target_kl}, resetting parameters ...')
            set_from_flat(thold)
            logger.log('done setting parameters')
            stop = True

        if mrew > best_rew:
            logger.log(f'reward: {best_rew} -> {mrew}. saving model.')
            best_rew = mrew
            model.save(f'{savepath}/best/')

        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        # Calculates if value function is a good predicator of the returns (ev > 1)
        # or if it's just worse than predicting nothing (ev =< 0)
        ev = explained_variance(values, returns)

        if tensorboard:
            fd = {
                pl_ph: lossvals[0] * pg_coef * rl_coef,
                vl_ph: lossvals[1] * vf_coef * rl_coef,
                pe_ph: lossvals[2] * ent_coef * rl_coef,
                ak_ph: lossvals[-2],
                cf_ph: lossvals[-1],
                stop_ph: int(stop),
                rew_ph: mrew,
            }

            if with_vae:
                fd.update({
                    ac_ph: np.reshape(actions, [-1, 1]),
                    ac_clip_ph: np.reshape(np.clip(actions, -2, 2), [-1, 1]),
                    rel_ph: alpha*re_l,
                    kll_ph: alpha*model.vae.beta*kl_l,
                })
                for i, kph in enumerate(klls_ph):
                    fd.update({kph: model.vae.beta*alpha*kl_ls[i]})

                summary = s.run(merged_, feed_dict=fd)
            else:
                fd.update({
                    ac_ph: np.reshape(actions, [-1, 1]),
                    ac_clip_ph: np.reshape(np.clip(actions, -2, 2), [-1, 1]),
                })
                summary = s.run(merged_, feed_dict=fd)

            fw.add_summary(summary, update*nbatch)

        if with_vae:
            csv.writeline(datetime.datetime.now().isoformat(), update, update * nbatch, fps, float(lossvals[2] * ent_coef * rl_coef),
                          float(lossvals[1] * vf_coef * rl_coef), float(lossvals[0] * pg_coef * rl_coef), float(ev),
                          float(mrew), float(lossvals[-2]), float(lossvals[-1]), int(early_stop and stop),
                          alpha*re_l, alpha*model.vae.beta*kl_l, *(model.vae.beta*kl_ls*alpha))
        else:
            csv.writeline(datetime.datetime.now().isoformat(), update, update * nbatch, fps, float(lossvals[2] * ent_coef * rl_coef),
                          float(lossvals[1] * vf_coef * rl_coef), float(lossvals[0] * pg_coef * rl_coef), float(ev), float(mrew),
                          float(lossvals[-2]), float(lossvals[-1]), int(early_stop and stop))

        if update % log_interval == 0 or update == 1:
            if plot_thetas:
                mus, _ = model.vae.encode(theta_frames)
                mus = np.reshape(mus, [200, 5])
                th_sum_eval = s.run(th_im_sum, feed_dict={mus_ph: np.moveaxis(mus,0,1)})
                fw.add_summary(th_sum_eval, update * nbatch)

                if with_vae:
                    sampled_obs = obs[np.random.choice(nbatch, 5)]

                    sampled_obs = np.expand_dims(np.moveaxis(sampled_obs, -1, 1), -1)
                    reconstructions = model.vae.reconstruct_stacked(sampled_obs)

                    vf_grad = s.run(vf_grad_op, feed_dict={model.vae.X: sampled_obs})
                    vf_grad = np.squeeze(np.asarray(vf_grad, dtype=np.float32))[:, 0, ...].copy()

                    # scale gradients to [0,1]
                    vf_grad += np.abs(np.min(vf_grad))
                    vf_grad *= (1/(np.abs(np.min(vf_grad)) + np.max(vf_grad)))

                    shw = []
                    for l in range(sampled_obs.shape[0]):
                        samo = np.repeat(sampled_obs[l, 0, ...], 3, axis=-1)
                        reco = np.repeat(np.asarray(reconstructions[l, ...], dtype=np.float32), 3, axis=-1)
                        lob = samo.copy()
                        lob[..., -1] += np.where(vf_grad[l] > np.mean(vf_grad[l]) * 1.1, vf_grad[l], np.zeros_like(vf_grad[l]))
                        cano = np.concatenate((samo, reco, lob), axis=1)

                        shw.append(cano)

                    shw = np.concatenate(shw, axis=0)

                    im_sum_eval = s.run(im_sum, feed_dict={img_ph: np.expand_dims(shw, axis=0)})
                    fw.add_summary(im_sum_eval, update * nbatch)

            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("stop", int(early_stop and stop))
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', mrew)
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()

            perc = ((update * nbatch) / total_timesteps) * 100
            steps2go = total_timesteps - (update * nbatch)
            secs2go = steps2go / fps
            min2go = secs2go / 60

            hrs = int(min2go // 60)
            mins = int(min2go) % 60
            print(colorize('ETA: {}h {}min | done {}% '.format(hrs, mins, int(perc)), color='cyan'))

        if update % save_interval == 0:
            model.save(savepath)

    model.save(savepath)
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



