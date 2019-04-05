import tensorflow as tf
import functools

from gym.spaces import Box

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

import numpy as np
from forkan.models import VAE


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
    def __init__(self, vae_name, k, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, microbatch_size=None):
        self.sess = sess = get_session()

        self.v = VAE(load_from=vae_name, network='pendulum', with_opt=False)

        self.vae_x = tf.placeholder(tf.float32, shape=(None, k,)+self.v.input_shape[1:], name='stacked-vae-input')
        disent_x = [self.vae_x[:, i, ...] for i in range(k)]
        self.obs_tensor = tf.concat(self.v.stack_encoder(disent_x), axis=1)

        model_ob_space = Box(low=-2, high=2, shape=(k*self.v.latent_dim, ), dtype=np.float32)
        self.t = 0

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess, ob_space=model_ob_space, observ_placeholder=self.obs_tensor)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess, ob_space=model_ob_space, observ_placeholder=self.obs_tensor)
            else:
                train_model = policy(microbatch_size, nsteps, sess, ob_space=model_ob_space, observ_placeholder=self.obs_tensor)

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
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model') + tf.trainable_variables('vae/encoder')
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

    def value(self, obs, **extra_feed):
        obs = np.expand_dims(np.moveaxis(obs, -1, 1), -1)
        feed_dict = {self.vae_x: obs}
        am = self.act_model

        for inpt_name, data in extra_feed.items():
            if inpt_name in am.__dict__.keys():
                inpt = am.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = am.adjust_shape(inpt, data)

        return np.asarray(am.sess.run([am.vf], feed_dict=feed_dict), dtype=np.float32)

    def step(self, obs, **extra_feed):
        obs = np.expand_dims(np.moveaxis(obs, -1, 1), -1)
        feed_dict = {self.vae_x: obs}
        am = self.act_model

        for inpt_name, data in extra_feed.items():
            if inpt_name in am.__dict__.keys():
                inpt = am.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = am.adjust_shape(inpt, data)

        return am.sess.run([am.action, am.vf, am.state, am.neglogp], feed_dict=feed_dict)

    def save(self, savepath):
        save_variables(savepath, tf.trainable_variables('ppo2_model'), sess=self.sess)
        self.v._save('retrain')

    def load(self, loadpath):
        save_variables(loadpath, tf.trainable_variables('ppo2_model'), sess=self.sess)
        self.v._load('retrain')

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        obs = np.expand_dims(np.moveaxis(obs, -1, 1), -1)

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.vae_x: obs,
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

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

