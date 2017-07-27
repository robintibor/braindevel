import numpy as np
from braindevel.analysis.create_amplitude_perturbation_corrs import load_exp_pred_fn
from braindevel.analysis.create_amplitude_perturbation_corrs import (
    create_batch_inputs_targets_amplitude_phase, 
    perturb_and_compute_covariances)
from braindevel.paper.amp_corrs import transform_to_corrs_preds_last


def create_perturbation_correlations(basename, with_square,
    with_square_cov, after_softmax, n_samples):
    exp, pred_fn = load_exp_pred_fn(basename, after_softmax=after_softmax)
    inputs, targets, amplitudes, phases = create_batch_inputs_targets_amplitude_phase(
        exp)
    batch_size = exp.iterator.batch_size
    all_orig_preds = np.array([pred_fn(batch_in) for batch_in in inputs])
    [all_covs, all_var_amps, all_var_preds] = perturb_and_compute_covariances(inputs, amplitudes, phases, 
        all_orig_preds, batch_size,
        pred_fn, n_samples, with_square, with_square_cov)
    all_corrs = transform_to_corrs_preds_last(all_covs, all_var_amps, all_var_preds)
    np.save(basename + ".all_corrs.npy", all_corrs)

if __name__ == "__main__":
    #basename = 'data/models/online/cnt/anla/with-highpass/3'
    #basename = 'data/models/online/cnt/hawe/with-highpass/17'
    #basename = 'data/models/online/cnt/lufi/with-highpass/1'
    #basename = 'data/models/online/cnt/sama/with-highpass/3'
    n_samples = 300
    with_square = False
    with_square_cov = False
    after_softmax = False
    create_perturbation_correlations(basename,
        with_square=with_square, with_square_cov=with_square_cov,
        after_softmax=after_softmax, n_samples=n_samples)
    