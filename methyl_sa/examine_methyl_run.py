from typing import Callable, Tuple

import pdb
import re

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

from pineappleflow.core.flyte.loaders.flyte_experiment_loader import FlyteExperimentLoader
from pineappleflow.core.flyte.loaders.flyte_inference_loader import FlyteInferenceLoader


_TRAINING_RUN_ID = "fda6c15a5789b4f3eabb"
_INFERENCE_RUN_ID = "f3bde335c82b448d488e"
_RNG = np.random.default_rng()
_NORM_TCPG_COUNT = 100_000
_N_REROLLS = 10


def plot_methyl_dists(data, png_prefix):
    sns.displot(data=data, x='methyl', hue='type', kind='ecdf')
    plt.savefig(png_prefix + f".methyl.dists.png", dpi=300)
    plt.close("all")


def simplify_pathological_type(typename):
    if typename is not None:
        if typename.startswith("Negative"):
            return "NEG"
        elif typename == "Adenocarcinoma":
            return "CRC"
        elif typename == "AA":
            return "AA"
    return None


def debug_context():
    eloader = FlyteExperimentLoader.from_run_id(_TRAINING_RUN_ID, domain="development", project="pineapple")
    e_pre = eloader.fold('train_set_final').pre_transformer_fold_holder.train
    e_post = eloader.fold('train_set_final').post_transformer_fold_holder.train
    # TODO(jsh): this is how babby NEG is form
    empre = e_pre[e_pre.features[0]]
    empost = e_post[e_post.features[0]]
    iloader = FlyteInferenceLoader.from_run_id(_INFERENCE_RUN_ID, domain="development", project="pineapple")
    i_pre = iloader.fold('metadata_balanced_kfold_stage_0').pre_transformer_matrix_holder
    i_post = iloader.fold('metadata_balanced_kfold_stage_0').post_transformer_matrix_holder
    impre = i_pre[i_pre.features[0]]
    impost = i_post[i_post.features[0]]

    pre_data = pd.DataFrame(index=impre.sample_ids)
    pre_types = [rmd.dataset.sample_metadata.pathologic_type for rmd in impre.row_metadata]
    pre_data['type'] = [simplify_pathological_type(x) for x in pre_types]
    pre_data['label'] = impre.y
    RELEVANT_REGION = 236
    CHOSEN_CUTOFF = 5
    full_slice = impre.x[:, RELEVANT_REGION, CHOSEN_CUTOFF, :]
    pre_data['methyl'] = full_slice[:, 0] / full_slice[:, 1]
    # plot_methyl_dists(pre_data, 'pre_transformer_chain')

    post_data = pd.DataFrame(index=impost.sample_ids)
    post_types = [rmd.dataset.sample_metadata.pathologic_type for rmd in impost.row_metadata]
    post_data['type'] = [simplify_pathological_type(x) for x in post_types]
    post_data['label'] = impost.y
    post_data['methyl'] = impost.x[:, 1]
    # plot_methyl_dists(post_data, 'post_transformer_chain')

    neg_mask = e_post[e_post.features[0]].y == 0
    mcpg = e_post[e_post.features[0]][neg_mask].x.sum(axis=0)[1]
    tcpg = e_post[e_post.features[0]][neg_mask].x.shape[0] * _NORM_TCPG_COUNT
    c = i_post[i_post.features[0]].x[:, 1].astype(int)
    h = _NORM_TCPG_COUNT - c

    pre_mcpg_tcpg_pairs = i_pre[i_pre.features[0]].x.sum(axis=0)
    pmtp = pre_mcpg_tcpg_pairs
    pre_flippers = list()
    for reg in range(pmtp.shape[0]):
        reg_slice = list()
        pre_flippers.append(reg_slice)
        for thresh in range(pmtp.shape[1]):
            flipper = sample_reflipper(*pmtp[reg, thresh])
            reg_slice.append(flipper)
    pre_flippers = np.array(pre_flippers)

    i_pre_mat = i_pre[i_pre.features[0]]
    assert i_pre_mat.shape[1:] == pmtp.shape

    perturbed_slices = list()
    for reg in range(pmtp.shape[0]):
        reg_sub = list()
        for thresh in range(pmtp.shape[1]):
            print(f"re-flipping region: {reg}, thresh: {thresh}")
            base_slice = i_pre_mat[:, reg, thresh, :].x
            pre_flipper = pre_flippers[reg, thresh]
            perturbations = list()
            for pair in base_slice:
                perturbed = pre_flipper(*pair, _N_REROLLS)
                perturbations.append(perturbed)
            perturbed_slice = np.concatenate(perturbations)
            reg_sub.append(perturbed_slice)
    perturbed_slices = np.array(perturbed_slices)
    pdb.set_trace()

    print("I guess we're done debugging!")


# p(c'|c) = p(C)*p(c|C)*p(c'|C) + p(H)*p(c|H)*p(c'|H)
# p(c'|h) = p(C)*p(h|C)*p(c'|C) + p(H)*p(h|H)*p(c'|H)

# p(c|H) = p_flipped
# p(c'|H) = p_flipped
# p(h|H) = 1 - p_flipped

# THE BIG ASSUMPTION
# p(c|C) = 1 - p_flipped
# p(c'|C) = 1 - p_flipped
# p(h|C) = p_flipped

# P(H) = 1 - p(C)


def sample_reflipper(mcpg_neg: int, tcpg_neg: int) -> Callable[[int, int], Tuple[int, int]]:
    p_flipped = mcpg_neg / tcpg_neg
    min_p_cancer = 1 / (10 * _NORM_TCPG_COUNT)
    p_cancer = np.concatenate([np.array([0]), np.geomspace(min_p_cancer, 1, 1000)])
    p_c_c = ((1-p_cancer) * p_flipped**2) +\
            (p_cancer * (1-p_flipped)**2)
    p_c_h = (p_cancer * p_flipped * (1-p_flipped)) +\
            ((1-p_cancer) * (1-p_flipped) * p_flipped)
    # p_c_h = p_flipped - p_flipped**2

    def reflip_func(c_m: int, h_m: int, n_resample: int) -> Tuple[int, int]:
        split_post_agg = list()
        # iterate over all ways to split c_m over the two source cases (c|c) and (c|h)
        from_c = np.array(range(0, c_m+1))
        from_h = c_m - from_c
        split_post = st.binom.pmf(from_c, c_m, p_c_c[:, np.newaxis]) * st.binom.pmf(from_h, h_m, p_c_h[:, np.newaxis])
        split_probs = split_post.sum(axis=1)
        mle_idx = np.argmax(split_probs)
        best_p_c_c = p_c_c[mle_idx]
        best_p_c_h = p_c_h[mle_idx]
        trials = c_m + h_m
        if np.isnan(best_p_c_c) or np.isnan(best_p_c_h):
            # if the negs had no data for this slice, leave it alone
            return np.tile([c_m, h_m], [n_resample, 1])
        else:
            c_perturbed = _RNG.binomial(c_m, best_p_c_c, n_resample) +\
                          _RNG.binomial(h_m, best_p_c_h, n_resample)
            h_perturbed = trials - c_perturbed
            return np.array([c_perturbed, h_perturbed]).T

    return reflip_func


def adjust_c_fracs(trials, p_flip, c_domain) -> np.ndarray: # [0, 1]
    c_rates = c_domain / trials
    return (c_rates * (1-p_flip)) + ((1-c_rates) * p_flip)


def adjust_measured_pair(c_m: int, h_m: int, mcpg_neg: int, tcpg_neg: int) -> Tuple[int, int]:
    trials = c_m + h_m
    p_flip = mcpg_neg / tcpg_neg
    domain = np.arange(0, trials+1)
    adjusted_c_fracs = adjust_c_fracs(trials, p_flip, domain)
    c_best = np.argmax(st.binom.pmf(c_m, trials, adjusted_c_fracs))
    h_best = trials - c_best
    return (c_best, h_best)


def adjust_measured_pair_jesse(c_m: int, h_m: int, mcpg_neg: int, tcpg_neg: int) -> Tuple[int, int]:
    def jesse(a: int, b: int, a0: int, b0: int) -> int:
        """grab the fraction closest to a/a+b from [a0->b0]/a0+b0 and return the index."""
        return np.argmin(np.abs(a / (a + b) - np.linspace(a0 / (a0 + b0), b0 / (a0 + b0), num=a + b + 1)))
    tmp = jesse(c_m, h_m, mcpg_neg, tcpg_neg - mcpg_neg)
    return tmp, c_m + h_m - tmp


def main():
    debug_context()


if __name__ == "__main__":
    main()
