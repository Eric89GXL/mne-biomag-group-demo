"""
=======================
Analysis for subject 13
=======================

Run the analysis.
"""
import os.path as op
import sys

import numpy as np

import mne

sys.path.append(op.join('..', '..', 'processing'))
from library.config import study_path, meg_dir, ylim, l_freq  # noqa: E402

###############################################################################
# Configuration

subjects_dir = op.join(study_path, 'subjects')

subject = "sub013"
subject_dir = op.join(meg_dir, subject)

###############################################################################
# Continuous data
raw_fname = op.join(study_path, 'ds117', subject, 'MEG', 'run_01_raw.fif')
raw_filt_fname = op.join(subject_dir,
                         'run_01_filt_sss_highpass-%sHz_raw.fif' % l_freq)
raw = mne.io.read_raw_fif(raw_fname)
raw_filt = mne.io.read_raw_fif(raw_filt_fname)

###############################################################################
# Filtering :ref:`sphx_glr_auto_scripts_04-python_filtering.py`.
raw.plot_psd(n_fft=8192, average=False, xscale='log', show=False)
raw_filt.plot_psd(n_fft=8192, average=False, xscale='log')

###############################################################################
# Events :ref:`sphx_glr_auto_scripts_02-extract_events.py`.
# Epochs :ref:`sphx_glr_auto_scripts_06-make_epochs.py`.
eve_fname = op.join(subject_dir, 'run_01_filt_sss-eve.fif')
epo_fname = op.join(subject_dir,
                    '%s_highpass-%sHz-epo.fif' % (subject, l_freq))

events = mne.read_events(eve_fname)
fig = mne.viz.plot_events(events, show=False)
fig.suptitle('Events from run 01')

epochs = mne.read_epochs(epo_fname)
epochs.plot_drop_log()

###############################################################################
# Evoked responses :ref:`sphx_glr_auto_scripts_07-make_evoked.py`
ave_fname = op.join(subject_dir,
                    '%s_highpass-%sHz-ave.fif' % (subject, l_freq))
evoked = mne.read_evokeds(ave_fname)

###############################################################################
# Faces
famous_evo, scrambled_evo, unfamiliar_evo, contrast_evo, faces_evo = evoked[:5]
faces_evo.plot(spatial_colors=True, gfp=True, ylim=ylim,
               window_title='Faces %s' % subject)

###############################################################################
# Famous
famous_evo.plot(spatial_colors=True, gfp=True, ylim=ylim,
                window_title='Famous %s' % subject)

###############################################################################
# Scrambled
scrambled_evo.plot(spatial_colors=True, gfp=True, ylim=ylim,
                   window_title='Scrambled %s' % subject)

###############################################################################
# Unfamiliar
unfamiliar_evo.plot(spatial_colors=True, gfp=True, ylim=ylim,
                    window_title='Unfamiliar %s' % subject)

###############################################################################
# Faces - scrambled
contrast_evo.plot(spatial_colors=True, gfp=True, ylim=ylim,
                  window_title='Faces - scrambled %s' % subject)

###############################################################################
# Topomaps
times = np.arange(0.05, 0.3, 0.025)
famous_evo.plot_topomap(times=times, title='Famous %s' % subject,
                        show=False)
scrambled_evo.plot_topomap(times=times, title='Scrambled %s' % subject,
                           show=False)
unfamiliar_evo.plot_topomap(times=times, title='Unfamiliar %s' % subject,
                            show=False)
contrast_evo.plot_topomap(times=times, title='Faces - scrambled %s' % subject,
                          show=True)

###############################################################################
# ICA
ica_fname = op.join(subject_dir, 'run_concat-ica.fif')
ica = mne.preprocessing.read_ica(ica_fname)
ica.plot_sources(raw_filt, show=True)

###############################################################################
# TFR :ref:`sphx_glr_auto_scripts_09-time_frequency.py`.
fpower = mne.time_frequency.read_tfrs(
    op.join(subject_dir, '%s-faces-tfr.h5' % subject))[0]
fitc = mne.time_frequency.read_tfrs(
    op.join(subject_dir, '%s-itc_faces-tfr.h5' % subject))[0]
spower = mne.time_frequency.read_tfrs(
    op.join(subject_dir, '%s-scrambled-tfr.h5' % subject))[0]
sitc = mne.time_frequency.read_tfrs(
    op.join(subject_dir, '%s-itc_scrambled-tfr.h5' % subject))[0]
channel = 'EEG065'
idx = [fpower.ch_names.index(channel)]
fpower.plot(idx, title='Faces power %s' % channel, baseline=(-0.1, 0.0),
            mode='logratio', show=False)
spower.plot(idx, title='Scrambled power %s' % channel, baseline=(-0.1, 0.0),
            mode='logratio', show=False)
fitc.plot(idx, title='Faces ITC %s' % channel, baseline=(-0.1, 0.0),
          mode='logratio', show=False)
sitc.plot(idx, title='Scrambled ITC %s' % channel, baseline=(-0.1, 0.0),
          mode='logratio')


###############################################################################
# Covariance :ref:`sphx_glr_auto_scripts_07-make_evoked.py`.
cov_fname = op.join(subject_dir,
                    '%s_highpass-%sHz-cov.fif' % (subject, l_freq))
cov = mne.read_cov(cov_fname)
mne.viz.plot_cov(cov, faces_evo.info)
faces_evo.copy().apply_baseline().plot_white(cov)

###############################################################################
# Trans
fname_trans = op.join(study_path, 'ds117', subject, 'MEG',
                      '%s-trans.fif' % subject)
bem = mne.read_bem_surfaces(op.join(subjects_dir, subject, 'bem',
                                    '%s-5120-bem.fif' % subject))
src = mne.read_source_spaces(
    op.join(subjects_dir, subject, 'bem', '%s-oct6-src.fif' % subject))
mne.viz.plot_alignment(famous_evo.info, fname_trans, subject=subject,
                       subjects_dir=subjects_dir, bem=bem, src=src,
                       surfaces='inner_skull')

###############################################################################
# Faces :ref:`sphx_glr_auto_scripts_13-make_inverse.py`.


def plot_stc(cond, figure=None):
    fname = op.join(subject_dir, 'mne_dSPM_inverse-%s' % cond)
    stc = mne.read_source_estimate(fname, subject).magnitude()
    brain = stc.plot(subject=subject, subjects_dir=subjects_dir, views=['ven'],
                     hemi='both', initial_time=0.17, time_unit='s',
                     figure=figure)
    return brain


brain_faces = plot_stc('faces', figure=1)

###############################################################################
# Faces - scrambled
brain_contrast = plot_stc('contrast', figure=2)

###############################################################################
# LCMV Faces - scrambled
fname = op.join(subject_dir, 'mne_LCMV_inverse-contrast')
stc = mne.read_source_estimate(fname, subject)
stc.plot(subject=subject, subjects_dir=subjects_dir, views=['ven'],
         hemi='both', initial_time=0.17, time_unit='s', figure=3)

###############################################################################
# BEM
mne.viz.plot_bem(subject, subjects_dir)
