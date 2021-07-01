from typing import Callable, Tuple

from collections import defaultdict
from copy import deepcopy
import pdb

# from matplotlib import pyplot as plt
# from matplotlib import cm
import joypy
import numpy as np
import pandas as pd

# from statsmodels.graphics.gofplots import qqplot
# import seaborn as sns
import scipy.stats as st

from calzone.models import (
    CalzoneDatasetMetadata,
)
from pineappleflow.core.artifacts.matrix import MatrixHolder, Matrix, RowMetadata
from pineappleflow.core.flyte.loaders.flyte_experiment_loader import (
    FlyteExperimentLoader,
)
from pineappleflow.core.flyte.loaders.flyte_inference_loader import FlyteInferenceLoader


_TRAINING_RUN_ID = "fda6c15a5789b4f3eabb"
_INFERENCE_RUN_ID = "f3bde335c82b448d488e"
_RNG = np.random.default_rng()
_NORM_TCPG_COUNT = 100_000
_N_REROLLS = 100
_PERTURBED_X_FILENAME = "perturbed_x.npy"


# Type for methyl noise sample reflipper functions
ReflipFunc = Callable[[int, int, int], Tuple[int, int]]


def build_loaders(
    train_id: str,
    inf_id: str,
) -> Tuple[
    FlyteExperimentLoader,
    MatrixHolder,
    MatrixHolder,
    FlyteExperimentLoader,
    MatrixHolder,
    MatrixHolder,
]:
    # set up FlyteLoaders
    eloader = FlyteExperimentLoader.from_run_id(
        train_id, domain="development", project="pineapple"
    )
    e_pre = eloader.fold("train_set_final").pre_transformer_fold_holder.train
    e_post = eloader.fold("train_set_final").post_transformer_fold_holder.train
    iloader = FlyteInferenceLoader.from_run_id(
        inf_id, domain="development", project="pineapple"
    )
    i_pre = iloader.fold(
        "metadata_balanced_kfold_stage_0"
    ).pre_transformer_matrix_holder
    i_post = iloader.fold(
        "metadata_balanced_kfold_stage_0"
    ).post_transformer_matrix_holder
    return eloader, e_pre, e_post, iloader, i_pre, i_post


def per_slice_sample_reflippers(base_matrix: Matrix) -> np.ndarray:
    """Build sample_reflippers for (reg,thresh) slices.

    returns:
        slice_flippers: np.ndarray(ReflipFunc, shape=(n_regions, n_settings))
    """
    neg_mask = base_matrix[base_matrix.features[0]].y == 0
    mcpg_tcpg_pairs = base_matrix[base_matrix.features[0]][neg_mask].x.sum(axis=0)
    pmtp = mcpg_tcpg_pairs
    slice_flippers = list()
    for reg in range(pmtp.shape[0]):
        reg_slice = list()
        slice_flippers.append(reg_slice)
        for thresh in range(pmtp.shape[1]):
            flipper = sample_reflipper(*pmtp[reg, thresh])
            reg_slice.append(flipper)
    slice_flippers = np.array(slice_flippers)
    return slice_flippers


def find_or_build_perturbed_x(
    base_matrix: Matrix, slice_flippers: np.ndarray
) -> np.ndarray:
    """Build new x data for base_matrix using provided per-slice reflippers

    returns:
        np.ndarray(int, shape=base_matrix.x.shape)"""
    base_feature = base_matrix[base_matrix.features[0]]
    assert base_feature.shape[1:3] == slice_flippers.shape

    try:
        slice_pert_x = np.load(_PERTURBED_X_FILENAME)
    except FileNotFoundError:
        perturbed_slices = list()
        for reg in range(slice_flippers.shape[0]):
            reg_sub = list()
            for thresh in range(slice_flippers.shape[1]):
                base_slice = base_feature[:, reg, thresh, :].x
                top_pair = base_slice[np.argmax(base_slice[:, 0])]
                print(f"re-flipping region: {reg}, thresh: {thresh}, top: {top_pair}")
                slice_flipper = slice_flippers[reg, thresh]
                slice_perturbations = list()
                for mcpg, tcpg in base_slice:
                    perturbed = slice_flipper(mcpg, tcpg - mcpg, _N_REROLLS)
                    slice_perturbations.append(perturbed)
                perturbed_slice = np.concatenate(slice_perturbations)
                reg_sub.append(perturbed_slice)
            perturbed_slices.append(reg_sub)
        perturbed_slices = np.array(perturbed_slices)
        slice_pert_x = np.transpose(perturbed_slices, (2, 0, 1, 3))
        np.save(_PERTURBED_X_FILENAME, slice_pert_x)
    return slice_pert_x


def build_new_metas(base_metas: np.ndarray, n_rerolls: int) -> np.ndarray:
    "expand each metadata to n_rerolls copies but with counter-tagged unique dsids"
    redundant_metas = np.repeat(base_metas, n_rerolls)
    dsid_counter = defaultdict(int)
    non_redundant_metas = list()
    for rmd in redundant_metas:
        udsid = f"{rmd.dataset.id}/{dsid_counter[rmd.dataset.id]}"
        dsid_counter[rmd.dataset.id] += 1
        udsmd = deepcopy(rmd.dataset.raw_dataset_metadata)
        udsmd["id"] = udsid
        urmdds = CalzoneDatasetMetadata(
            raw_dataset_metadata=udsmd,
            raw_sample_metadata=rmd.dataset.raw_sample_metadata,
            raw_tube_metadata=rmd.dataset.raw_tube_metadata,
            raw_quiche_metadata=rmd.dataset.raw_quiche_metadata,
            raw_plasmaqc_metadata=rmd.dataset.raw_plasmaqc_metadata,
            raw_study_metadata=rmd.dataset.raw_study_metadata,
        )
        urmd = RowMetadata(label=rmd.label, source=rmd.source, dataset=urmdds)
        non_redundant_metas.append(urmd)
    expanded_row_metas = np.array(non_redundant_metas)
    return expanded_row_metas


def sample_reflipper(mcpg_neg: int, tcpg_neg: int) -> ReflipFunc:
    """
    Logical Grounding:
        p(c'|c) = p(C)*p(c|C)*p(c'|C) + p(H)*p(c|H)*p(c'|H)
        p(c'|h) = p(C)*p(h|C)*p(c'|C) + p(H)*p(h|H)*p(c'|H)

        p(c|H) = p_flipped
        p(c'|H) = p_flipped
        p(h|H) = 1 - p_flipped

        THE BIG ASSUMPTION
        p(c|C) = 1 - p_flipped
        p(c'|C) = 1 - p_flipped
        p(h|C) = p_flipped

        P(H) = 1 - p(C)
    """
    p_flipped = mcpg_neg / tcpg_neg
    min_p_cancer = 1 / (10 * _NORM_TCPG_COUNT)
    p_cancer = np.concatenate([np.array([0]), np.geomspace(min_p_cancer, 1, 1000)])
    p_c_c = ((1 - p_cancer) * p_flipped ** 2) + (p_cancer * (1 - p_flipped) ** 2)
    p_c_c = p_c_c[:, np.newaxis]
    p_c_h = (p_cancer * p_flipped * (1 - p_flipped)) + (
        (1 - p_cancer) * (1 - p_flipped) * p_flipped
    )
    p_c_h = p_c_h[:, np.newaxis]

    # TODO(jsh): this should return mcpg/tcpg, and (ONLY THEN) outer functions should behave accordingly
    def reflip_func(c_m: int, h_m: int, n_resample: int) -> Tuple[int, int]:
        # iterate over all ways to split c_m over the two source cases (c|c) and (c|h)
        from_c = np.array(range(0, c_m + 1))
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
            c_perturbed = _RNG.binomial(c_m, best_p_c_c, n_resample) + _RNG.binomial(
                h_m, best_p_c_h, n_resample
            )
            h_perturbed = trials - c_perturbed
            return np.array([c_perturbed, h_perturbed]).T

    return reflip_func


def pre_poisson_counts(loader: FlyteExperimentLoader, pre_mat: Matrix) -> pd.DataFrame:
    """Apply transformer chain and extract pre-poisson count data.

    args:
        loader: FlyteExperimentLoader containing the fitted transformer chain
        pre_mat: pre-transformer-chain count matrix
    returns:
        count_frame:
            "unscaled_mcpg": mid-transformer-chain un-standardized mcpg (just before "poisson-outlier" standard)
            "unscaled_tcpg": corresponding imputed un-standardized tcpgs
            "hmfc": the scaled output score ("fragments per 100_000")
            "scale_factor": the scale_factor for converting unscaled_mcpg to post_mat mcpg
            "y": the dataset y-label
    """
    fold = loader.fold("train_set_final")
    trans_mat = pre_mat
    transformers = fold.feature(pre_mat.name).fitted_transformers
    intermediate_mats = list()
    for transformer in transformers:
        trans_mat = transformer.transform_inference(trans_mat)
        intermediate_mats.append(trans_mat)
    poisson_pos = next(
        i
        for i in range(len(transformers))
        if "PoissonOutlier" in transformers[i].spec["name"]
    )
    pre_poisson = poisson_pos - 1
    unscaled_mcpg = pd.Series(
        np.rint(intermediate_mats[pre_poisson].x[:, :, 0].sum(axis=1)),
        index=intermediate_mats[pre_poisson].dataset_ids,
        name="unscaled_mcpg",
    )
    unscaled_tcpg = pd.Series(
        np.rint(intermediate_mats[pre_poisson].x[:, :, 1].sum(axis=1)),
        index=intermediate_mats[pre_poisson].dataset_ids,
        name="unscaled_tcpg",
    )
    hmfc = pd.Series(
        trans_mat.x[:, 1],
        index=trans_mat.dataset_ids,
        name="hmfc",
    )
    y = pd.Series(
        intermediate_mats[pre_poisson].y,
        index=intermediate_mats[pre_poisson].dataset_ids,
        name="y",
    )
    count_frame = pd.DataFrame(
        [unscaled_mcpg, unscaled_tcpg, hmfc, y],
    ).T.sort_index()
    count_frame['base_dsid'] = [x.split('/')[0] for x in count_frame.index]
    count_frame['scale_factor'] = (_NORM_TCPG_COUNT / count_frame.unscaled_tcpg)
    count_frame['unscaled_tcpg'] = count_frame['unscaled_tcpg'].astype(int)
    count_frame["unscaled_mcpg"] = count_frame["unscaled_mcpg"].astype(int)
    count_frame["y"] = count_frame["y"].astype(int)
    return count_frame


def plot_methyl_dists(data, png_prefix):
    sns.displot(data=data, x="methyl", hue="type", kind="ecdf")
    plt.savefig( f"{png_prefix}.methyl.dists.png", dpi=300)
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


def main():
    eloader, e_pre, _e_post, iloader, i_pre, _i_post = build_loaders(
        _TRAINING_RUN_ID, _INFERENCE_RUN_ID
    )
    assert len(e_pre.features) == 1
    assert len(_e_post.features) == 1
    assert len(i_pre.features) == 1
    assert len(_i_post.features) == 1

    e_pre_mat = e_pre[e_pre.features[0]]
    train_count_frame = pre_poisson_counts(eloader, e_pre_mat)
    neg_count_frame = train_count_frame.loc[train_count_frame.y == 0]
    mcpg = neg_count_frame.unscaled_mcpg.sum()
    tcpg = neg_count_frame.unscaled_tcpg.sum()
    post_flipper = sample_reflipper(mcpg, tcpg)

    i_pre_mat = i_pre[i_pre.features[0]]
    base_count_frame = pre_poisson_counts(eloader, i_pre_mat)

    _NUM_POST_PERT_TRIALS = 10
    perturbations = defaultdict(list)
    for _, row in base_count_frame.iterrows():
        mcpg = row.unscaled_mcpg
        tcpg = row.unscaled_tcpg
        for i in range(_NUM_POST_PERT_TRIALS):
            perturbed = post_flipper(mcpg, tcpg - mcpg, _N_REROLLS)
            hmfcs = perturbed[:, 0] * row.scale_factor
            perturbations[f"trial_{i}"].append(hmfcs)
    trial_cols = dict()
    for i in range(_NUM_POST_PERT_TRIALS):
        trial_cols[f"trial_{i}"] = np.concatenate(perturbations[f"trial_{i}"])
    # TODO(jsh): These clearly need to be saved out to disk!
    post_pert_trial_frame = pd.DataFrame(trial_cols)
    post_pert_trial_frame["base_dsid"] = np.repeat(base_count_frame.index, _N_REROLLS)

    slice_reflippers = per_slice_sample_reflippers(e_pre)
    perturbed_pre_x = find_or_build_perturbed_x(i_pre, slice_reflippers)
    perturbed_pre_x[..., 1] += perturbed_pre_x[..., 0]
    perturbed_pre_rmds = build_new_metas(i_pre_mat.row_metadata, _N_REROLLS)
    pert_i_pre_mat = i_pre_mat.replace_x_and_axis_metadata(
        perturbed_pre_x, [perturbed_pre_rmds, *i_pre_mat.axis_metadata[1:]]
    )
    pre_pert_count_frame = pre_poisson_counts(eloader, pert_i_pre_mat)

    post_pert_trial_frame["pre_pert"] = pre_pert_count_frame.reset_index().hmfc
    for dsid, group in post_pert_trial_frame.groupby("base_dsid"):
        joypy.joyplot(group, legend=False)
        plt.savefig(f"post_trials.{dsid}.joyplot.png", dpi=300)

    pdb.set_trace()

    return


if __name__ == "__main__":
    main()
