import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns;

sns.set()
import glob2
import argparse


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
                                                                          mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


wd = os.getcwd()
policy_dirs = sys.argv[1:]
# Load all data.
data = {}

paths = [wd + '/' + p for p in policy_dirs]
for curr_path in paths:
    if not os.path.isdir(curr_path):
        continue

    results = load_results(os.path.join(curr_path, 'progress.csv'))
    if not results:
        print('skipping {}'.format(curr_path))
        continue
    print('loading {} ({})'.format(curr_path, len(results['epoch'])))
    with open(os.path.join(curr_path, 'params.json'), 'r') as f:
        params = json.load(f)

    success_rate = np.array(results['test/success_rate'])
    epoch = np.array(results['epoch']) + 1
    env_id = params['env_name']
    replay_strategy = params['replay_strategy']

    if replay_strategy == 'future':
        config = 'her'
    else:
        config = 'ddpg'
    if 'Dense' in env_id:
        config += '-dense'
    else:
        config += '-sparse'
    env_id = env_id.replace('Dense', '')

    # Process and smooth data.
    assert success_rate.shape == epoch.shape
    x = epoch
    y = success_rate
    assert x.shape == y.shape

    if env_id not in data:
        data[env_id] = {}
    if config not in data[env_id]:
        data[env_id][config] = []
    data[env_id][config].append((x, y))

# This plot is for comparison in the same env only. check if we have different envs
if len(data.keys()) > 1:
    print('only one env is supported for comparison')
    sys.exit(1)

env = list(data.keys())[0]

print('exporting {}'.format(env))
plt.clf()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for config in sorted(data[env].keys()):
    xs, ys = zip(*data[env][config])


    for i in range(len(xs)):
        x, y = [], []
        x = [np.array(xs[i].copy())]
        y = [np.array(ys[i].copy())]

        xt = tuple(x)
        yt = tuple(y)
        xt, yt = pad(xt), pad(yt)
        assert xt.shape == yt.shape

        plt.plot(xt[0], np.nanmedian(yt, axis=0), label=config)
        # plt.fill_between(x[i], np.nanpercentile(y, 25, axis=0), np.nanpercentile(y, 75, axis=0), alpha=0.25)

        # plt.plot(xs[i], ys[i], label=policy_dirs[i])

axes = plt.gca()
axes.set_ylim([0, 0.25])

plt.title(env)
plt.xlabel('Epoch')
plt.ylabel('Median Success Rate')
plt.legend()
plt.savefig(os.path.join(wd, 'fig_{}.png'.format(env_id)))

plt.show()
