import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

F1 = os.environ['HOME'] + '/shapes.npz'
F2 = os.environ['HOME'] + '/shapes_without.npz'

w = np.load(F1, encoding='latin1')
wo = np.load(F2, encoding='latin1')

xw = w['x']
yw = w['y']

xwo = wo['x']
ywo = wo['y']

cut = len(xwo) - 1

xw = xw[:cut]
yw = yw[:cut]

plt.plot(xw, yw, label='with touch')
plt.plot(xwo, ywo, label='without touch')

plt.xlabel('Epoch')
plt.ylabel('Median Success Rate')
plt.legend()

plt.savefig('{}/new_general.png'.format(os.environ['HOME']))

plt.show()