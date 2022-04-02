import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
import torch
import torch.nn as nn
from fgn import ROOT
from fgn.dataset.offlinedataset import RotAugCategoryGrasp
from fgn.dataset.pyr_onlinerenderv2 import uniform_quaternion
from fgn.model.discrete_cond_flow import *
from fgn.model.eqflow.vn_flow import *
from fgn.utils.grasp_eval import GraspEvaluation
from fgn.utils.utils import MLP, Transform, TupleSeq, pv_cannonical_gripper
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--scale_fn", type=str, default="sigmoidsoftplus")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--prefix", type=str, default="BookBootleMugAugRot")
    args = parser.parse_args()
    return args


def transrotvec2T(transrotvec):
    trans = transrotvec[:3]
    rotvec = transrotvec[3:]

    rotmat = Rotation.from_rotvec(rotvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotmat
    T[:3, 3] = trans

    return T


def plot_pair(savedir: Path, title: str, prior_z, grasp_z):

    df = pd.DataFrame()
    num_data = prior_z.shape[0]
    for i in range(7):
        if i == 0:
            df["class"] = ["prior"] * num_data + ["grasp"] * num_data
        else:
            df[f"dim{i}"] = np.hstack([prior_z[:, i - 1], grasp_z[:, i - 1]])

    print(df.head())
    g = sns.PairGrid(df, hue="class")
    # g.map_upper(sns.kdeplot)
    g.map_upper(sns.scatterplot, alpha=0.2)

    # g.map_lower(sns.kdeplot)
    g.map_lower(sns.scatterplot, alpha=0.2)

    g.map_diag(sns.histplot, kde=True)

    # g.add_legend(title=title)
    g.add_legend()

    if not savedir.exists():
        savedir.mkdir(parents=True)

    plt.savefig(f"{savedir}/{title}.png", dpi=300)
    # plt.show()


def main(args):
    print(args.train)
    device = torch.device("cuda")

    fnames = [
        "Book_8daa30dd38bd5f5cd5bd8e7415322d0f_0.014203253615105615",
        "Bottle_a86d587f38569fdf394a7890920ef7fd_0.022478530465288797",
        "Mug_414772162ef70ec29109ad7f9c200d62_0.0008565924686492948",
    ]
    model_dir = ROOT / f"experiments/{args.prefix}"
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    model_name = f"best_model.pt"

    dataset = RotAugCategoryGrasp(fnames, rotrepr="rotmat")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
    )

    data_dim = 6
    num_flow = 10
    num_hidden = 512

    # create new instance when it called
    def net():
        return TupleSeq(
            MLP([data_dim // 2 + dataset.num_object + 9, num_hidden, num_hidden]),
            nn.Linear(num_hidden, data_dim // 2 * 2),
        )

    transforms = []

    for i in range(num_flow):
        transforms.append(
            ConditionalAffineCoupling(net(), args.scale_fn, nn.Identity())
        )
        transforms.append(ActNorm(data_dim))
        if i != num_flow - 1:
            transforms.append(Reverse(data_dim))

    model = ConditionalFlow(StandardNormal((data_dim,)), transforms).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_epoch = args.num_epoch
    # num_epoch = 1

    ###########
    ## Train ##
    ###########
    if args.train:
        model.train()
        print("Training...")
        l = 0.0
        criteria = np.inf
        for epoch in range(num_epoch):
            l = 0.0
            for i, data in enumerate(train_loader):
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = -model(x, y)[1].mean()
                loss.backward()
                optimizer.step()
                l += loss.detach().cpu().item()

                if l < criteria:
                    criteria = l
                    # print("model save!")
                    torch.save(model.state_dict(), str(model_dir / model_name))
                print(
                    "Epoch: {}/{}, Loglik: {:.3f}".format(
                        epoch + 1, num_epoch, l / (i + 1)
                    ),
                    end="\r",
                )
            print("")

    ############
    ## Sample ##
    ############

    print("Sampling...")

    model.load_state_dict(torch.load(str(model_dir / model_name)))
    model.eval()
    num_samples = 1000
    # object_idx = 0
    num_rot = 10
    grasp_mesh = pv_cannonical_gripper(np.eye(4))

    quatitative_data = []

    for object_idx in range(dataset.num_object):
        print(object_idx, "---------------------------------")
        for rot_idx in range(num_rot):
            save_dir = (
                ROOT
                / f"results/3obj_uniformpose/sample{num_samples}_obj{object_idx}_rot{rot_idx}"
            )

            quat = uniform_quaternion(1).reshape(-1)
            action_transform = Transform.from_xyzquat(np.zeros(3), quat)

            original_samples = dataset.datas[object_idx][
                np.random.randint(0, len(dataset.datas[object_idx]), (num_samples,))
            ]
            rel_samples = []

            for i in range(num_samples):
                tmp = (
                    action_transform * Transform.from_xyzrotvec(original_samples[i])
                ).as_xyzrotvec()
                rel_samples.append(tmp)

            rel_samples = np.vstack(rel_samples)

            y = torch.zeros(num_samples, dataset.num_object + 9, device=device)

            y[:, object_idx] = 1.0
            y[:, dataset.num_object :] = (
                torch.from_numpy(action_transform.rotation.as_matrix().reshape(-1))
                .to(torch.float)
                .to(device)
            )

            # s = time.time()
            gen_samples = model.sample(num_samples, y).detach().cpu().numpy()
            # print(f"Sampling takes {time.time() - s:0.4f}")

            z_prior = model.base_dist.sample(num_samples).detach().cpu().numpy()
            z_grasp = (
                model.forward(torch.from_numpy(rel_samples).float().to(device), y)[0]
                .detach()
                .cpu()
                .numpy()
            )

            true_grasp = Transform.from_xyzrotvec(rel_samples)
            positive_grasp = Transform.from_xyzrotvec(gen_samples)
            object_mesh = dataset.pv_meshes[object_idx].transform(
                action_transform.as_matrix(), inplace=False
            )

            to_grasp_center = Transform.from_xyzquat(
                np.array([0, 0, 0.112]), np.array([0, 0, 0, 1.0])
            )
            grasp_eval = GraspEvaluation(
                true_grasp * to_grasp_center,
                positive_grasp * to_grasp_center,
                0.025,
                np.deg2rad(30.0),
            )

            # print('Plotting Latent Space...')
            # grasp_eval.visualize_latent_space(z_prior, z_grasp)
            print("Plotting Data Space...")
            grasp_eval.visualize_data_space(
                object_mesh, True, str(save_dir / "data_trans_space.png")
            )
            for percent in [0.1, 0.3, 0.5, 1.0]:
                precision, coverage, exp_coverage = grasp_eval.pr_cov_expcov_k(percent)

                print(
                    f"K:{int(percent*100):03d}% | precision: {precision:0.2f} | coverage: {coverage:0.2f} | exp_coverage: {exp_coverage:0.2f}"
                )
                quatitative_data.append(
                    [
                        object_idx,
                        rot_idx,
                        int(percent * 100),
                        precision,
                        coverage,
                        exp_coverage,
                    ]
                )
            print("=======")
            continue
            print("Plotting Latent Space...")

            plot_pair(save_dir, "latent_space", z_prior, z_grasp)

            print("Plotting Data Space...")
            plotter = pv.Plotter(
                shape=(1, 2), off_screen=not args.viz, window_size=[1280, 1024]
            )
            # real
            plotter.subplot(0, 0)
            plotter.add_text("Real")
            plotter.add_mesh(
                dataset.pv_meshes[object_idx].transform(
                    action_transform.as_matrix(), inplace=False
                ),
                color=[0.7, 0.7, 0.7],
            )

            for i in range(num_samples):
                plotter.add_mesh(
                    grasp_mesh.transform(transrotvec2T(rel_samples[i]), inplace=False),
                    color=[0, 1, 0],
                    opacity=0.2,
                )

            # gen
            plotter.subplot(0, 1)
            plotter.add_text("Gen")
            plotter.add_mesh(
                dataset.pv_meshes[object_idx].transform(
                    action_transform.as_matrix(), inplace=False
                ),
                color=[0.7, 0.7, 0.7],
            )

            for i in range(num_samples):
                plotter.add_mesh(
                    grasp_mesh.transform(transrotvec2T(gen_samples[i]), inplace=False),
                    color=[0, 1, 0],
                    opacity=0.2,
                )

            plotter.link_views()

            if args.viz:
                plotter.show(screenshot=True)
                pil_image = Image.fromarray(plotter.image)
                pil_image.save(str(save_dir / "data_space.png"))
            else:
                plotter.screenshot(str(save_dir / "data_space.png"))
    quatitative_df = pd.DataFrame(
        quatitative_data,
        columns=[
            "object_id",
            "rot_id",
            "percent",
            "precision",
            "coverage",
            "exp_coverage",
        ],
    )
    quatitative_df.to_csv("result.csv")


if __name__ == "__main__":
    main(parser())
