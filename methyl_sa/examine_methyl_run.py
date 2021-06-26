from typing import Callable, Tuple

from collections import defaultdict
from copy import deepcopy
import pdb
import re

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as st
import seaborn as sns

from calzone.models import (
    CalzoneDatasetMetadata,
)
from pineappleflow.core.artifacts.matrix import MatrixHolder, Matrix, RowMetadata
from pineappleflow.core.flyte.loaders.flyte_experiment_loader import FlyteExperimentLoader
from pineappleflow.core.flyte.loaders.flyte_inference_loader import FlyteInferenceLoader


_TRAINING_RUN_ID = "fda6c15a5789b4f3eabb"
_INFERENCE_RUN_ID = "f3bde335c82b448d488e"
_RNG = np.random.default_rng()
_NORM_TCPG_COUNT = 100_000
_N_REROLLS = 100


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


def FIRST_PART():
    pass


def debug_context():
    eloader = FlyteExperimentLoader.from_run_id(_TRAINING_RUN_ID, domain="development", project="pineapple")
    e_pre = eloader.fold('train_set_final').pre_transformer_fold_holder.train
    e_post = eloader.fold('train_set_final').post_transformer_fold_holder.train
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

    post_neg_mask = e_post[e_post.features[0]].y == 0
    mcpg = e_post[e_post.features[0]][post_neg_mask].x.sum(axis=0)[1]
    tcpg = e_post[e_post.features[0]][post_neg_mask].x.shape[0] * _NORM_TCPG_COUNT
    c = i_post[i_post.features[0]].x[:, 1].astype(int)
    h = _NORM_TCPG_COUNT - c
    post_flipper = sample_reflipper(mcpg, tcpg)
    post_perturbations = list()
    for mcpg, tcpg in zip(c, h):
        perturbed = post_flipper(mcpg, tcpg-mcpg, _N_REROLLS)
        post_perturbations.append(perturbed)
    post_pert_x = np.concatenate(post_perturbations)

    pre_neg_mask = e_pre[e_pre.features[0]].y == 0
    pre_mcpg_tcpg_pairs = e_pre[e_pre.features[0]][pre_neg_mask].x.sum(axis=0)
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

    # DEBUG
    # pmtp = pmtp[:2, :2]
    # DEBUG
    try:
        pre_pert_x = np.load("pert_x.npy")
    except FileNotFoundError:
        perturbed_slices = list()
        for reg in range(pmtp.shape[0]):
            reg_sub = list()
            for thresh in range(pmtp.shape[1]):
                base_slice = i_pre_mat[:, reg, thresh, :].x
                top_pair = base_slice[np.argmax(base_slice[:, 0])]
                print(f"re-flipping region: {reg}, thresh: {thresh}, top: {top_pair}")
                pre_flipper = pre_flippers[reg, thresh]
                pre_perturbations = list()
                for mcpg, tcpg in base_slice:
                    perturbed = pre_flipper(mcpg, tcpg-mcpg, _N_REROLLS)
                    pre_perturbations.append(perturbed)
                perturbed_slice = np.concatenate(pre_perturbations)
                reg_sub.append(perturbed_slice)
            perturbed_slices.append(reg_sub)
        perturbed_slices = np.array(perturbed_slices)
        pre_pert_x = np.transpose(perturbed_slices, (2, 0, 1, 3))
        np.save("pert_x.npy", pre_pert_x)

    redundant_pert_pre_row_metas = np.repeat(i_pre_mat.row_metadata, _N_REROLLS)

    dsid_counter = defaultdict(int)
    pert_pre_row_meta_list = list()
    for rmd in redundant_pert_pre_row_metas:
        udsid = f"{rmd.dataset.id}/{dsid_counter[rmd.dataset.id]}"
        dsid_counter[rmd.dataset.id] += 1
        udsmd = deepcopy(rmd.dataset.raw_dataset_metadata)
        udsmd['id'] = udsid
        urmdds = CalzoneDatasetMetadata(
            raw_dataset_metadata = udsmd,
            raw_sample_metadata=rmd.dataset.raw_sample_metadata,
            raw_tube_metadata=rmd.dataset.raw_tube_metadata,
            raw_quiche_metadata=rmd.dataset.raw_quiche_metadata,
            raw_plasmaqc_metadata=rmd.dataset.raw_plasmaqc_metadata,
            raw_study_metadata=rmd.dataset.raw_study_metadata,
        )
        urmd = RowMetadata(label=rmd.label, source=rmd.source, dataset=urmdds)
        pert_pre_row_meta_list.append(urmd)
    pert_pre_row_metas = np.array(pert_pre_row_meta_list)
    assert len(i_pre.features) == 1
    pert_i_pre_mat = i_pre_mat.replace_x_and_axis_metadata(
        pre_pert_x,
        [
            pert_pre_row_metas,
            *i_pre_mat.axis_metadata[1:]
        ]
    )

    efold = eloader.fold("train_set_final")
    etransformers  = efold.feature(pert_i_pre_mat.name).fitted_transformers
    pert_i_pre_trans_mat = pert_i_pre_mat
    intermediate_mats = list()
    for transformer in etransformers:
        pert_i_pre_trans_mat = transformer.transform_inference(pert_i_pre_trans_mat)
        intermediate_mats.append(pert_i_pre_trans_mat)

    mid_frame = pd.DataFrame(
        intermediate_mats[2].x[:, :, 0].sum(axis=1),
        index=intermediate_mats[2].dataset_ids,
        columns=["raw"]
    ).sort_index()
    end_frame = pd.DataFrame(
        pert_i_pre_trans_mat.x[:, 1],
        index=pert_i_pre_trans_mat.dataset_ids,
        columns=["scaled"]
    ).sort_index()
    pert_i_pre_trans_frame = pd.DataFrame(index=end_frame.index)
    pert_i_pre_trans_frame["raw"] = mid_frame.raw
    pert_i_pre_trans_frame["scaled"] = end_frame.scaled
    factor = (100_000 * pert_i_pre_trans_frame.raw) / pert_i_pre_trans_frame.scaled
    pert_i_pre_trans_frame["tcpg"] = 100000
    pert_i_pre_trans_frame.loc[mask, "tcpg"] = factor[mask]

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
    p_c_c = p_c_c[:, np.newaxis]
    p_c_h = (p_cancer * p_flipped * (1-p_flipped)) +\
            ((1-p_cancer) * (1-p_flipped) * p_flipped)
    p_c_h = p_c_h[:, np.newaxis]

    # memolog = dict()
    def reflip_func(c_m: int, h_m: int, n_resample: int) -> Tuple[int, int]:
        split_post_agg = list()
        # iterate over all ways to split c_m over the two source cases (c|c) and (c|h)
        from_c = np.array(range(0, c_m+1))
        from_h = c_m - from_c
        split_post = st.binom.pmf(from_c, c_m, p_c_c) * st.binom.pmf(from_h, h_m, p_c_h)
        split_probs = split_post.sum(axis=1)
        mle_idx = np.argmax(split_probs)
        best_p_c_c = p_c_c[mle_idx, 0]
        best_p_c_h = p_c_h[mle_idx, 0]
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


def main():
    debug_context()


if __name__ == "__main__":
    main()
