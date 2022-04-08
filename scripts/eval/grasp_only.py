import argparse
from pathlib import Path
import torch
from gsl import ROOT
from gsl.dataset import read_all
from gsl.models import MODEL_ZOO
from gsl.utils.eval import GraspEvaluation
from gsl.utils.geometry import Transform

from scipy.spatial.distance import cdist
import numpy as np
import time
import pandas as pd
# checkpoint
# - FLOW_scalesigmoidsoftplus_N4
# - VAE_z2_kl1.0

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=lambda p: Path(p).absolute())
    parser.add_argument('--num_sample', type=int, default=int(1e+4))
    parser.add_argument('--trans_threshold', type=float, default=0.015) #15 mm
    parser.add_argument('--quat_threshold', type=float, default=np.pi/180.0 * 1.0) # 1 deg

    parser.add_argument('--ds', action='store_true', help="data space visulization")
    parser.add_argument('--ls', action='store_true', help="latent space visulization")
    parser.add_argument('--q', action='store_true', help="compare precision coverage")

    args = parser.parse_args()
    return args


def main():
    args = parser()

    # return
    seeds_model_ckpt = list(args.model_dir.glob("**/*.ckpt"))
    
    object_name = str(args.model_dir.parent.name)
    method = str(args.model_dir.name).split("_")[0] # VAE or FLOW

    print(f"Object: {object_name}")
    print(f"Method: {method}")
    print(f"num seed: {len(seeds_model_ckpt)}")
    quatitative_data = []


    # for each seed evaulate grasp
    for seed, model_ckpt in enumerate(seeds_model_ckpt):
        print(model_ckpt.name)
        model = MODEL_ZOO[method].load_from_checkpoint(model_ckpt).cuda()
        model.eval()


        # Data sapce setup
        positive_grasps = None
        if method == "VAE" or method == "GAN":
            transrotmatflat = model.sample_grasp(args.num_sample, None)
            teye = torch.eye(4).unsqueeze(0).repeat(args.num_sample, 1, 1)
            teye[..., :3,  3] = transrotmatflat[:, :3]
            teye[..., :3, :3] = transrotmatflat[:, 3:].reshape(-1, 3, 3)
            
            positive_grasps = Transform.from_matrix(teye.detach().cpu().numpy())

        elif method == "FLOW":
            transrotvec = model.sample_grasp(args.num_sample, None).detach().cpu().numpy()
            positive_grasps = Transform.from_xyzrotvec(transrotvec)


        poses, pv_mesh, pv_tex = read_all(object_name)
        gt_posquat = poses[np.random.choice(poses.shape[0], args.num_sample)]
        true_grasps = Transform.from_xyzquat(gt_posquat)



        grasp_eval = GraspEvaluation(true_grasps, positive_grasps, args.trans_threshold, args.trans_threshold)
        if args.q:
            print(f"{seed} precision converage analysis")
            for percent in [0.001, 0.01, 0.1, 0.5, 1.0]:
                num_samples, precision, coverage, exp_coverage= grasp_eval.pr_cov_expcov_k(percent)
                data_row = [seed, num_samples, precision, coverage, exp_coverage]
                quatitative_data.append(data_row)

        # latent space setup
        if args.ls:
            grasp_z = model.grasp_latent(torch.from_numpy(gt_posquat).float().cuda(), None).detach().cpu().numpy()
            prior_z = np.random.randn(*grasp_z.shape)
            grasp_eval.visualize_latent_space(prior_z, grasp_z)

        # visualize data space
        if args.ds:
            grasp_eval.visualize_data_space(pv_mesh, pv_tex, True)
            assert args.num_sample <= 1000, "less than 1000 is recommanded for visualize rotation"
            grasp_eval.visualize_data_space(pv_mesh, pv_tex, False)


    # store in csv at model_dir
    if args.q:
        quatitative_df = pd.DataFrame(quatitative_data, columns=["seed", "number of samples", "precision_rate (%)", "coverage_rate (%)", "exp_coverage"])
        quatitative_df.to_csv(str(args.model_dir /"result.csv"))

if __name__ == "__main__":
    main()
