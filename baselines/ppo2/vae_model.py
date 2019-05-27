import tensorflow as tf

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from gym.spaces import Box

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

import numpy as np
from forkan.models import RetrainVAE


class VAEModel(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, k, policy, ob_space, ac_space, nbatch_act, nbatch_train, savepath, env, vae_params, rl_coef,
                v_net, nsteps, ent_coef, vf_coef, pg_coef, max_grad_norm, microbatch_size=None, with_kl=False):
        self.sess = sess = get_session()

        v_in_shape = env.observation_space.shape[:-1] + (1,)

        # it's important to pass the session we are using in ppo
        self.vae = RetrainVAE(savepath, v_in_shape, network=v_net, sess=self.sess, **vae_params)

        retrain = 'init_from' in vae_params

        model_ob_space = Box(low=-2, high=2, shape=(k*self.vae.latent_dim, ), dtype=np.float32)
        self.t = 0
        self.k = k

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess, ob_space=model_ob_space, observ_placeholder=self.vae.U)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess, ob_space=model_ob_space, observ_placeholder=self.vae.U)
            else:
                train_model = policy(microbatch_size, nsteps, sess, ob_space=model_ob_space, observ_placeholder=self.vae.U)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_coef * pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        if retrain:
            params = tf.trainable_variables('ppo2_model') + tf.trainable_variables('vae/encoder')
            if with_kl:
                print('adding KL to loss')
                loss += (self.vae.beta * self.vae.kl_loss)
        else:
            print(f'using joint loss with rl_coef {rl_coef}')
            params = tf.trainable_variables()
            loss = rl_coef * loss + self.vae.vae_loss

        print('#params ', len(params))
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model

        self.initial_state = act_model.initial_state
        self.sess = sess

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

        # we have to load variables again after running the initializer
        if retrain:
            self.vae.load()

    def value(self, obs, **extra_feed):
        obs = np.expand_dims(np.moveaxis(obs, -1, 1), -1)
        feed_dict = {self.vae.X: obs}
        am = self.act_model

        # add additional data to feed dict
        for inpt_name, data in extra_feed.items():
            if inpt_name in am.__dict__.keys():
                inpt = am.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = am.adjust_shape(inpt, data)

        return np.asarray(am.sess.run([am.vf], feed_dict=feed_dict), dtype=np.float32)

    def step(self, obs, **extra_feed):
        obs = np.expand_dims(np.moveaxis(obs, -1, 1), -1)
        feed_dict = {self.vae.X: obs}
        am = self.act_model

        # add additional data to feed dict
        for inpt_name, data in extra_feed.items():
            if inpt_name in am.__dict__.keys():
                inpt = am.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = am.adjust_shape(inpt, data)

        return am.sess.run([am.action, am.vf, am.state, am.neglogp], feed_dict=feed_dict)

    def step_xhat(self, obs, **extra_feed):
        obs = np.expand_dims(np.moveaxis(obs, -1, 1), -1)
        feed_dict = {self.vae.X: obs}
        am = self.act_model

        # add additional data to feed dict
        for inpt_name, data in extra_feed.items():
            if inpt_name in am.__dict__.keys():
                inpt = am.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = am.adjust_shape(inpt, data)

        return am.sess.run([am.action, self.vae.Xhat, am.vf, self.vae.kl_loss, self.vae.re_loss, self.vae.vae_loss], feed_dict=feed_dict), obs

    def save(self, savepath):
        print('saving model ... ')
        save_variables(f'{savepath}/model', tf.trainable_variables('ppo2_model'), sess=self.sess)
        self.vae.save()

    def load(self, loadpath):
        print('loading model ...')
        load_variables(f'{loadpath}/model', tf.trainable_variables('ppo2_model'), sess=self.sess)
        self.vae.load()

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        obs = np.expand_dims(np.moveaxis(obs, -1, 1), -1)
        assert np.max(obs) <= 1, 'observations need to be normalized!'

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.vae.X: obs,
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        res = self.sess.run(
            self.stats_list + [self.vae.re_loss, self.vae.kl_loss, self._train_op],
            td_map
        )[:-1]

        kl_losses = res[-1]
        # mean losses
        re_loss = np.mean(res[-2])
        kl_loss = self.vae.beta * np.sum(kl_losses)

        return res[:-2], kl_losses, re_loss, kl_loss

        """ This again is for debugging. """
        #
        # import matplotlib.pyplot as plt
        #
        # res = self.sess.run(
        #     self.stats_list + [self.vae.U, self._train_op],
        #     td_map
        # )[:-1]
        #
        # mus = res[-1]
        # print('here ', mus.shape, obs.shape)
        #
        #
        # for i in range(10):
        #     fig, axes = plt.subplots(2, 3, figsize=(12, 10))
        #
        #     for j in range(self.k):
        #         axes[0, j].imshow(np.squeeze(obs[i, j, ...]))
        #         axes[0, j].set_title(f'k={j}')
        #
        #         print(mus[i][self.vae.latent_dim*j:self.vae.latent_dim*(j+1)].shape)
        #         axes[1, j].bar(np.arange(5), mus[i][self.vae.latent_dim*j:self.vae.latent_dim*(j+1)])
        #         axes[1, j].set_title(f'k={j}')
        #
        #     fig.tight_layout()
        #     fig.subplots_adjust(hspace=0.88)
        #
        #     plt.show()
        # exit()
        return res[:-1]

