import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
from gsl import ROOT

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', type=lambda p: Path(p).absolute())
    parser.add_argument('--save', action='store_true', help="save plot")
    
    args = parser.parse_args()
    return args

def main():
    args = parser()

    csvs = list(args.checkpoint_dir.glob("**/*.csv"))
    dfs = []
    for csv in csvs:
        method = csv.parent.name
        df = pd.read_csv(csv)
        length = len(df["seed"])
        df["method"] = pd.Series([method]*length, index=df.index)

        dfs.append(df)
    
    dfs = pd.concat(dfs, ignore_index=True)
    # print(dfs)
    object_name = csvs[0].parent.parent.name

    save_dir = ROOT / f"results/{object_name}/FLOW_VAE_GAN/"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    sns.lineplot(data=dfs, x="number of samples", y="coverage_rate (%)", hue="method")
    if args.save:
        plt.savefig(f"{save_dir}/cov_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    sns.lineplot(data=dfs, x="number of samples", y="precision_rate (%)", hue="method")
    if args.save:
        plt.savefig(f"{save_dir}/pre_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    sns.lineplot(data=dfs, x="number of samples", y="exp_coverage", hue="method")
    plt.ylabel("average shortest path")
    if args.save:
        plt.savefig(f"{save_dir}/asp_sample.png", dpi=300)
        plt.cla()
    else:
        plt.show()

    sns.lineplot(data=dfs, x="coverage_rate (%)", y="precision_rate (%)", hue="method")
    if args.save:
        plt.savefig(f"{save_dir}/cov_pr.png", dpi=300)
        plt.cla()
    else:
        plt.show()
    
if __name__ == "__main__":
    main()
