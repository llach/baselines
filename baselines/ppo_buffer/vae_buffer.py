import numpy as np
import random


class VAEBuffer(object):
    def __init__(self, size):
        """Create VAE buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, ret, acts, vals, nlps, reclosses):
        m_rec = np.median(reclosses)

        obs = obs[reclosses > m_rec]
        ret = ret[reclosses > m_rec]
        acts = acts[reclosses > m_rec]
        vals = vals[reclosses > m_rec]
        nlps = nlps[reclosses > m_rec]

        for i in range(obs.shape[0]):
            data = (obs[i], ret[i], acts[i], vals[i], nlps[i])

            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, returns, actions, values, neglogps = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, ret, acts, vals, nlps = data

            obses.append(np.array(obs, copy=False))
            returns.append(ret)
            actions.append(np.array(acts, copy=False))
            values.append(vals)
            neglogps.append(nlps)
        return np.array(obses), np.array(returns), np.array(actions), np.array(values), np.array(neglogps)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)