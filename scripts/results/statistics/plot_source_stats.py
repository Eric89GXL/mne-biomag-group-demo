"""
=============================================
Spatio-temporal source-space statistics (MEG)
=============================================

Clustering in source space.
"""
from functools import partial
import os.path as op
import sys

import numpy as np
from scipy import stats

import mne
from mne import spatial_src_connectivity
from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p

sys.path.append(op.join('..', '..', 'processing'))
from library.config import (meg_dir, subjects_dir, fsaverage_vertices,
                            exclude_subjects, N_JOBS, l_freq, random_state)  # noqa: E402

contrast = list()
tmin, tmax = 0.1, 0.8
tstep = None
for subject_id in range(1, 20):
    if subject_id in exclude_subjects:
        continue
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    data_path = op.join(meg_dir, subject)
    stc = mne.read_source_estimate(
        op.join(data_path, 'mne_dSPM_inverse_morph_highpass-%sHz-contrast'
                % (l_freq,)))
    assert stc.data.min() < 0
    assert stc.data.max() > 0
    if tstep is None:
        tstep = stc.tstep
    assert tstep == stc.tstep
    contrast.append(stc.crop(tmin, tmax).data.T)

###############################################################################
# Set up our contrast and threshold-free cluster enhancement (TFCE) parameters:

X = np.array(contrast, float)
fsaverage_src = mne.read_source_spaces(
    op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-5-src.fif'))
connectivity = spatial_src_connectivity(fsaverage_src)
p_thresholds = np.array([0.001, 0.0001])
t_thresholds = -stats.distributions.t.ppf(p_thresholds / 2., len(X) - 1)
threshold = dict(start=t_thresholds[0], step=np.diff(t_thresholds)[0])

###############################################################################
# Here we could do an exact test with ``n_permutations=2**(len(X)-1)``,
# i.e. 32768 permutations, but this would take a long time. For speed and
# simplicity we'll do 1024 (which is also the default).

n_permutations = 100
stat_fun = partial(ttest_1samp_no_p, sigma=1e-3)
#T_obs, clusters, cluster_p_values, H0 = clu = \
#    spatio_temporal_cluster_1samp_test(
#        X, connectivity=connectivity, n_jobs=N_JOBS, threshold=threshold,
#        stat_fun=stat_fun, buffer_size=None, seed=random_state,
#        n_permutations=n_permutations, verbose=True)

###############################################################################
# Visualize the results by collapsing along the time dimension:

t_obs, p_obs = stats.ttest_1samp(X, 0)
durations = (np.reshape(p_obs, X.shape[1:]) < 0.05) * np.sign(t_obs)
durations = durations.sum(axis=0, keepdims=True).T * tstep * 1000
stc_all_cluster_vis = mne.SourceEstimate(
    durations, fsaverage_vertices, 0., tstep=tstep, subject='fsaverage')
pos_lims = [50, 100, 500]
brain = stc_all_cluster_vis.plot(
    hemi='both', subjects_dir=subjects_dir,
    time_label='Duration significant (ms)', views='ven',
    clim=dict(pos_lims=pos_lims, kind='value'), size=(1000, 1000),
    background='white', foreground='black')
brain.save_image(op.join('..', 'figures', 'source_stats_highpass-%sHz.png'
                         % (l_freq,)))
