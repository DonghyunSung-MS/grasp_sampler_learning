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


def main():
    args = parser()
    # params = {'legend.fontsize': 8}
    # plt.rcParams.update(params)
    csvs = list(args.checkpoint_dir.glob("**/*.csv"))
    dfs = []
    for csv in csvs:
        method = csv.parent.name
        df = pd.read_csv(csv)
        length = len(df["seed"])
        df["method"] = pd.Series([method] * length, index=df.index)

        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    # return
    object_names = args.checkpoint_dir.name

    save_dir = ROOT / f"results/{object_names}/FLOW_VAE/"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    g = sns.FacetGrid(dfs, col="object name", hue="method")
    g.map(sns.lineplot, "number of samples", "coverage rate")
    g.add_legend()
    if args.save:
        plt.savefig(f"{save_dir}/cov_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    g = sns.FacetGrid(dfs, col="object name", hue="method")
    g.map(sns.lineplot, "number of samples", "precision rate")
    g.add_legend()
    if args.save:
        plt.savefig(f"{save_dir}/pre_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    g = sns.FacetGrid(dfs, col="object name", hue="method")
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
