import argparse
import datetime
from pathlib import Path

import numpy as np
import roma
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from gsl import ROOT
from gsl.dataset import GraspOnly, RotAugCategoryGrasp
from gsl.models import FlowGraspNet
from gsl.utils.geometry import (torch_transform2transquat,
                                torch_transform2transrotvec)


def parser():
    parser = argparse.ArgumentParser()
    # model param
    parser.add_argument("--num_flow", type=int, default=8)
    parser.add_argument("--scale_fn", type=str, default="exp")
    # train param
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=10000)
    # data param
    parser.add_argument("--data", type=str)
    parser.add_argument("--num_cluster", type=int, default=500)
    parser.add_argument(
        "--type", type=str, default="grasp", choices=["grasp", "categoryposegrasp"]
    )
    parser.add_argument(
        "--prefix", type=str, default=datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
    )

    args = parser.parse_args()
    return args


def get_model(args, fnames):

    if args.type == "grasp":
        return FlowGraspNet(6, 0, args.num_flow, args.scale_fn)

    elif args.type == "categoryposegrasp":

        # 9 for rotation matrix flatten 4 object category
        return FlowGraspNet(6, 9 + len(fnames), args.num_flow, args.scale_fn)


def get_dataloader(args, fnames):

    if args.type == "grasp":
        return DataLoader(GraspOnly(fnames[0], args.num_cluster), args.batch_size, True)
    elif args.type == "categoryposegrasp":
        return DataLoader(
            RotAugCategoryGrasp(fnames, args.num_cluster, "rotmat"),
            args.batch_size,
            True,
        )


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = parser()

    with open(args.data, "r") as f:
        fnames = f.read().splitlines()
    print(fnames)

    config = OmegaConf.create(vars(args))
    config.fnames = fnames
    config.model_name = "test.pt"

    save_dir = ROOT / f"checkpoints/{args.prefix}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    print("Saving Config")
    with open(save_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    dloader = get_dataloader(args, fnames)
    model = get_model(args, fnames)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    ###########
    ## Train ##
    ###########
    model.train()
    print("Training...")
    l = 0.0
    criteria = np.inf
    for epoch in range(args.num_epoch):
        l = 0.0
        for i, data in enumerate(dloader):

            if args.type == "grasp":
                x = data["x"]
                x = x.to(device)
                c = None
            elif args.type == "categoryposegrasp":
                x = data["x"]
                x = x.to(device)
                c = data["c"]
                c = c.to(device)

            optimizer.zero_grad()
            loss = -model(torch_transform2transrotvec(x), c)[1].mean()
            loss.backward()
            optimizer.step()

            l += loss.detach().cpu().item()
            if l < criteria:
                criteria = l
                print("model save!")
                torch.save(model.state_dict(), str(save_dir / config.model_name))
            print(
                "Epoch: {}/{}, Loglik: {:.3f}".format(
                    epoch + 1, args.num_epoch, l / (i + 1)
                ),
                end="\r",
            )
        print("")


if __name__ == "__main__":
    main()
