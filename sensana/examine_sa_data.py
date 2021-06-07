import os.path
import pdb
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

from pineappleflow.core.flyte.loaders.flyte_inference_loader import FlyteInferenceLoader


_RUN_ID_LIST = [
    "f928d7c02d42347c5a66",
    "fffbf936cfbc64b39b46",
    "f176780398b3343348c6",
    "f3beaec3f24ac4b25ae2",
    "f476dce3ef801420584f",
    "f2d34df609b0d45b0aab",
    "fd3dd14c1437a4e39a83",
    "f6f7d5df66a3742009b5",
    "fe2f5c8e9f0d74d2985c",
    "f51d01b1311f948459f4",
    "febc87d7c316540ed8c0",
    "f0082389367664dd58d4",
    "ff844bd75b6d74c7185e",
    "f9d0e907e7e3f4b168c3",
    "f5e6e73fc32f84517985",
    "fa1c9502fd02d4c39afc",
    "fd07fc8e0f7c842d394d",
    "f25cab46a5b4b4473bb8",
    "fffa9943dd05e4816b17",
    "f194cdaa4980543e7a63",
]


def simplify_pathological_type(typename):
    if typename.startswith("Negative"):
        return "NEG"
    elif typename == "Adenocarcinoma":
        return "CRC"
    elif typename == "AA":
        return "AA"
    else:
        raise ValueError("unknown pathological type!")


def create_sa_frames(run_id):
    loader = FlyteInferenceLoader.from_run_id(run_id, domain="development", project="pineapple")
    mh = loader.inference_matrix_holder
    pred = loader.aggregated_model_result.predictions
    fullname = mh[mh.features[0]].dataset_ids

    def get_parts(x):
        return filter(None, re.split(r"[\[\]]", x))

    dsid, ptag, subidx = list(zip(*[get_parts(x) for x in fullname]))
    row_metas = mh[mh.features[0]].row_metadata
    truth = [x.dataset.sample_metadata.pathologic_type for x in row_metas]
    truth = map(simplify_pathological_type, truth)
    sa_frame = pd.DataFrame(
        {
            "dsid": dsid,
            "ptag": ptag,
            "subidx": subidx,
            "pred": pred,
            "fullname": fullname,
            "truth": truth
        }
    )
    basecall_map = dict()
    for idx, row in sa_frame.loc[sa_frame.ptag == "unperturbed"].iterrows():
        basecall_map[row.dsid] = row.pred
    sa_frame["basecall"] = sa_frame.dsid.map(basecall_map)
    sa_frame["score"] = loader.aggregated_model_result.scores[:, 1]
    post_transformation_holder = loader.fold('train_set_final').post_transformer_matrix_holder
    butter_keys = [x for x in post_transformation_holder.features if "butter" in x]
    assert len(butter_keys) == 1
    butter_key = butter_keys[0]
    butter_data = post_transformation_holder[butter_key].x
    log_butter = np.log10(butter_data + 1).astype('int')
    two_dim_features = [f for f in mh.features if len(mh[f].shape) == 2]
    x_merged = np.hstack([mh[f].x for f in two_dim_features] + [log_butter])
    base_columns = [get_column_name(c) for f in two_dim_features for c in mh[f].column_metadata] + ["loghmf"]
    unique_columns = clean_columns(base_columns)
    big_frame = pd.DataFrame(data=x_merged, columns=unique_columns)
    sa_frame["loghmf"] = big_frame.loghmf
    sa_frame["cutoff"] = loader.aggregated_model_result.classification_thresholds[0]
    return big_frame, sa_frame


def clean_columns(columns):
    col_tracker = dict()
    unique_cols = list()
    for c in columns:
        if c not in col_tracker:
            col_tracker[c] = 0
            unique_cols.append(c)
        else:
            col_tracker[c] += 1
            unique_cols.append(f"{c}_{col_tracker[c]}")
    return np.array(unique_cols)


def get_column_name(column_metadata):
    try:
        return column_metadata['Assay']
    except ValueError:
        # Add options to the "try:" block until we don't end up here.
        pdb.set_trace()


def accumulate_frames(run_ids):
    sa_frame_header = None
    big_frame_header = None
    sa_frame_blocks = list()
    big_frame_blocks = list()
    for run_id in run_ids:
        big_frame, sa_frame = create_sa_frames(run_id)
        if sa_frame_header is None:
            sa_frame_header = sa_frame.loc[sa_frame.ptag == "unperturbed"]
        if big_frame_header is None:
            big_frame_header = big_frame.loc[sa_frame.ptag == "unperturbed"]
        sa_frame_blocks.append(sa_frame.loc[sa_frame.ptag != "unperturbed"])
        big_frame_blocks.append(big_frame.loc[sa_frame.ptag != "unperturbed"])
    agg_frame = pd.concat([sa_frame_header] + sa_frame_blocks)
    big_frame = pd.concat([big_frame_header] + big_frame_blocks)
    return big_frame, agg_frame


def _percent_positive(x):
    fractions = x.pred.sum() / len(x)
    percents = 100 * fractions
    return percents


def build_heatmap_frame(agg_frame):
    tidy = agg_frame.reset_index().groupby(["ptag", "dsid"]).apply(_percent_positive)
    rectangular = tidy.unstack().T
    return rectangular


def draw_heatmap(agg_frame, heat_frame, png_filename):
    hmf_palette = sns.color_palette("Set2", len(agg_frame.loghmf.unique()))
    hmf_lut = dict(zip(sorted(agg_frame.loghmf.unique()), hmf_palette))
    hmf_color_map = agg_frame.loc[agg_frame.ptag == "unperturbed"][["dsid", "loghmf"]].set_index('dsid')
    hmf_colors = hmf_color_map.loc[heat_frame.index].loghmf.map(hmf_lut)
    truth_palette = sns.color_palette("Dark2", len(agg_frame.truth.unique()))
    truth_lut = dict(zip(agg_frame.truth.unique(), truth_palette))
    truth_color_map = agg_frame.loc[agg_frame.ptag == "unperturbed"][["dsid", "truth"]].set_index('dsid')
    truth_colors = truth_color_map.loc[heat_frame.index].truth.map(truth_lut)
    sns.clustermap(heat_frame, cmap="vlag", row_colors=[hmf_colors, truth_colors])
    plt.savefig(png_filename, dpi=300)
    plt.close("all")


def scatter_score_vs_pospercent(big_frame, agg_frame, png_filename):
    agg_frame["delta"] = agg_frame.score - agg_frame.cutoff
    pospercent = 100 * agg_frame.groupby('dsid').pred.sum() / agg_frame.groupby('dsid').pred.count()
    plotframe = agg_frame.loc[agg_frame.ptag == "unperturbed", ["dsid", "delta"]].copy()
    plotframe.set_index("dsid", inplace=True)
    plotframe['pospercent'] = pospercent
    sns.scatterplot(data=plotframe, x="delta", y="pospercent", s=5)
    plt.xlim(-0.01, 0.01)
    plt.savefig(png_filename, dpi=300)
    plt.close("all")


def plot_variation_hists(big_frame, agg_frame, png_prefix):
    idx_list = list()
    for x in agg_frame.ptag:
        if "1feat" in x:
            idx_list.append(int(x[x.find("(") + 1 : x.find(",")]))
    idx = np.array(list(set(idx_list)))
    for i in idx:
        rows = big_frame.loc[agg_frame.ptag.str.contains(str(i))]
        col = big_frame.columns[i]
        sns.displot(rows.iloc[:, i])
        plt.savefig(png_prefix + f".{col}.{i}.png", dpi=300)
        plt.close("all")


def debug_context(big_frame, agg_frame, heat_frame):
    idx_list = list()
    for x in agg_frame.ptag:
        if "1feat" in x:
            idx_list.append(int(x[x.find("(") + 1 : x.find(",")]))
    idx = np.array(list(set(idx_list)))
    loader = FlyteInferenceLoader.from_run_id(_RUN_ID_LIST[0], domain="development", project="pineapple")
    mh = loader.fold('train_set_final').post_transformer_matrix_holder
    m0 = mh[mh.features[0]]
    m1 = mh[mh.features[1]]
    clif = loader.experiment_loader.fold('train_set_final').fitted_model_recipe.classifier
    pdb.set_trace()
    print("I guess we're done debugging!")


def main():
    store = pd.HDFStore('sensana.store.h5')
    # if True:
    if ('agg_frame' not in store) or ('big_frame' not in store):
        big_frame, agg_frame = accumulate_frames(_RUN_ID_LIST)
        store['agg_frame'] = agg_frame
        store['big_frame'] = big_frame
    else:
        big_frame = store['big_frame']
        agg_frame = store['agg_frame']

    # if True:
    if 'heat_frame' not in store:
        heat_frame = build_heatmap_frame(agg_frame)
        store['heat_frame'] = heat_frame
    else:
        heat_frame = store['heat_frame']

    # DEBUG
    debug_context(big_frame, agg_frame, heat_frame)
    return
    # DEBUG
    draw_heatmap(agg_frame, heat_frame, "sensana.clustermap.png")
    scatter_score_vs_pospercent(big_frame, agg_frame, "sensana.score.fragility.png")
    plot_variation_hists(big_frame, agg_frame, "sensana.1feat.variation")


if __name__ == "__main__":
    main()


def test_create_sa_frame():
    sa_frame = create_sa_frames(_RUN_ID)
    orig_rows = sa_frame.loc[sa_frame.ptag == "unperturbed"]
    assert (orig_rows.pred == orig_rows.basecall).all()


def test_accumulate_frames():
    _, agg_frame = accumulate_frames(_RUN_ID_LIST)
    orig_rows = agg_frame.loc[agg_frame.ptag == "unperturbed"]
    assert len(orig_rows) == 144
    assert "night" == "day"
