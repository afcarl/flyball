from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import gzip
import cPickle as pickle

from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.stats import whiten, cov

import autoregressive.models as models
import autoregressive.distributions as distributions

from load import load

np.random.seed(0)


### load

data = load()
ndim = data.shape[1]

### model

Nmax = 10
affine = True
nlags = 3

model = models.ARWeakLimitStickyHDPHMM(
    alpha=4., gamma=4.,
    kappa=1e6,
    init_state_distn='uniform',
    obs_distns=[
        distributions.AutoRegression(
            nu_0=ndim+1,
            S_0=np.eye(ndim),
            M_0=np.zeros((ndim,ndim*nlags+affine)),
            K_0=np.eye(ndim*nlags+affine),
            affine=affine)
        for state in range(Nmax)],
)

model.add_data(data)

### inference

# for itr in progprint_xrange(500):
#     model.resample_model()

# model.plot_stateseq(model.states_list[0], plot_slice=slice(6000,8000))
# plt.show()


### animation

from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip

# plt.set_cmap('terrain')
plot_slice = slice(5000,8000)

fig = plt.figure(figsize=(20,3))
model.plot_stateseq(model.states_list[0],draw=False,plot_slice=plot_slice)
ax = plt.gca()

def make_frame_mpl(t):
    model.resample_model()
    model.plot_stateseq(model.states_list[0],ax=ax,update=True,draw=False,plot_slice=plot_slice)
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame_mpl, duration=15)
animation.write_videofile('gibbs.mp4',fps=15)
