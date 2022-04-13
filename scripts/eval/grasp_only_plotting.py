import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gsl import ROOT


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=lambda p: Path(p).absolute())
    parser.add_argument("--save", action="store_true", help="save plot")
    parser.add_argument("--run_name", type=str, help="Save dir name")

    args = parser.parse_args()
    return args


"""
plot structure
col object
method colors set 3

"""


def main():
    args = parser()
    csvs = list(args.checkpoint_dir.glob("**/*.csv"))
    dfs = []
    for csv in csvs:
        method = csv.parent.name
        df = pd.read_csv(csv)
        length = len(df["seed"])
        df["method"] = pd.Series([method] * length, index=df.index)

        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)
    # print(dfs)
    object_name = csvs[0].parent.parent.name

    save_dir = ROOT / f"results/{object_name}/ALOTOFCOMPARISON/"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    hue_order = sorted(list(set(dfs["method"])))
    # set color

    vae_df = dfs.where(dfs["method"].str.contains("VAE"))
    gan_df = dfs.where(dfs["method"].str.contains("GAN"))
    flow_df = dfs.where(dfs["method"].str.contains("FLOW"))

    g = sns.FacetGrid(vae_df, hue="method", height=6, aspect=2, palette="tab10")
    g.map(sns.lineplot, "number of samples", "coverage rate", ci=None)
    g.add_legend()

    if args.save:
        plt.savefig(f"{save_dir}/vae_cov_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    g = sns.FacetGrid(gan_df, hue="method", height=6, aspect=2, palette="tab10")
    g.map(sns.lineplot, "number of samples", "coverage rate", ci=None)
    g.add_legend()

    if args.save:
        plt.savefig(f"{save_dir}/gan_cov_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    g = sns.FacetGrid(flow_df, hue="method", height=6, aspect=2, palette="tab10")
    g.map(sns.lineplot, "number of samples", "coverage rate", ci=None)
    g.add_legend()

    if args.save:
        plt.savefig(f"{save_dir}/flow_cov_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    return
    # Get best avg Method
    cond = (
        (dfs["method"] == "VAE_z4_kl0.01")
        | (dfs["method"] == "FLOW_scalesigmoidsoftplus_N8")
        | (dfs["method"] == "GAN_z2")
    )

    masked_df = dfs.where(cond)

    g = sns.FacetGrid(
        masked_df, hue="method", palette="tab10", legend_out=False, height=6
    )
    g.map(sns.lineplot, "number of samples", "coverage rate")
    g.add_legend()
    if args.save:
        plt.savefig(f"{save_dir}/cov_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    g = sns.FacetGrid(
        masked_df, hue="method", palette="tab10", legend_out=False, height=6
    )
    g.map(sns.lineplot, "number of samples", "average shortest path")
    g.add_legend()
    if args.save:
        plt.savefig(f"{save_dir}/asp_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    # sns.lineplot(data=dfs, x="coverage rate", y="precision rate", hue="method")
    # plt.title(object_name)
    # if args.save:
    #     plt.savefig(f"{save_dir}/cov_pr.png", dpi=300)
    #     plt.cla()
    # else:
    #     plt.show()


if __name__ == "__main__":
    main()
