"""
=================================
14. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import os.path as op

import numpy as np

import mne
from mne.parallel import parallel_func

from library.config import (meg_dir, subjects_dir, N_JOBS, smooth,
                            fsaverage_vertices, exclude_subjects, l_freq)


def morph_stc(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    data_path = op.join(meg_dir, subject)

    # Contrast in sensor space projected to source space
    condition = 'contrast'
    fname_in = op.join(data_path, 'mne_dSPM_inverse_highpass-%sHz-%s')
    fname_out = op.join(data_path, 'mne_dSPM_inverse_morph_highpass-%sHz-%s')
    stc = mne.read_source_estimate(fname_in % (l_freq, condition), subject)
    morph_mat = mne.compute_morph_matrix(
        subject, 'fsaverage', stc.vertices, fsaverage_vertices, smooth,
        subjects_dir=subjects_dir, warn=False)
    contrast = stc.morph_precomputed('fsaverage', fsaverage_vertices,
                                     morph_mat, subject)
    contrast.save(fname_out % (l_freq, condition))
    # Contrast of source space absolute values
    contrast_abs = [mne.read_source_estimate(fname_in % (l_freq, condition))
                    for condition in ['faces_eq', 'scrambled_eq']]
    contrast_abs = [c.morph_precomputed('fsaverage', fsaverage_vertices,
                                        morph_mat, subject)
                    for c in contrast_abs]
    for c, condition in zip(contrast_abs, ['faces_eq', 'scrambled_eq']):
        c.save(fname_out % (l_freq, condition))
    contrast_abs = abs(contrast_abs[0]) - abs(contrast_abs[1])
    contrast_abs.save(fname_out % (l_freq, 'contrast_eq_abs'))
    return [contrast, contrast_abs]


parallel, run_func, _ = parallel_func(morph_stc, n_jobs=N_JOBS)
stcs = parallel(run_func(subject_id) for subject_id in range(1, 20))
stcs = [stc for stc, subject_id in zip(stcs, range(1, 20))
        if subject_id not in exclude_subjects]
vertices, tmin, tstep = stcs[0][0].vertices, stcs[0][0].tmin, stcs[0][0].tstep

data = np.average([s[0].data for s in stcs], axis=0)
stc = mne.SourceEstimate(data, vertices, tmin, tstep, 'fsaverage')
stc.save(op.join(meg_dir, 'contrast-average_highpass-%sHz' % l_freq))

data = np.average([s[1].data for s in stcs], axis=0)
stc = mne.SourceEstimate(data, vertices, tmin, tstep, 'fsaverage')
stc.save(op.join(meg_dir, 'contrast_eq_abs-average_highpass-%sHz' % l_freq))
