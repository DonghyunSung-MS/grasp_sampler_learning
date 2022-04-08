from pathlib import Path
from typing import List

import h5py
import numpy as np
# from sklearn.cluster import KMeans
import pyvista as pv
import torch
import trimesh
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from gsl import ROOT
from gsl.utils.geometry import Transform, uniform_quaternion


def read_grasps(grasp_path: Path, num_train, num_test):

    data = h5py.File(grasp_path, "r")
    # scale = data["object_scale"][()]
    poses = data["poses"][()]  # xyz, xyzw
    success = data["quality_flex_success"][()]
    mesh_file = data["object"][()]
    print(type(mesh_file))
    if isinstance(mesh_file, bytes):
        mesh_file = mesh_file.decode("UTF-8")
    # print(scuc)
    rate = (success == 1).sum() / success.shape[0]
    # return
    object_name = mesh_file.split("/")[0]

    print(object_name, poses.shape, success.shape, rate)

    sample_grasps = poses[
        np.random.choice(poses.shape[0], num_train + num_test, replace=True)
    ]
    train_grasp = sample_grasps[:num_train]
    test_grasp = sample_grasps[num_train:]

    # sample from train and test to check distribution shift effect
    eval_grasp = np.vstack(
        [
            train_grasp[np.random.choice(num_train, num_test // 2, replace=False)],
            test_grasp[np.random.choice(num_test, num_test // 2, replace=False)],
        ]
    )

    return train_grasp, eval_grasp, test_grasp


def read_all(object_name):
    pv_mesh = pv.read(ROOT / f"data/models/{object_name}/textured.obj")
    pv_tex = pv.read_texture(ROOT / f"data/models/{object_name}/texture_map.png")
    data = h5py.File(ROOT / f"data/grasps/{object_name}/grasps.h5", "r")
    poses = data["poses"][()]  # xyz, xyzw
    return poses, pv_mesh, pv_tex


class GraspOnly(Dataset):
    """
    cog centered grasp translation distribution with only positive grasp
    """

    def __init__(self, grasp: np.array) -> None:
        # grasp np.array(N, 7) xyz xyzw
        self.datas = grasp.astype(np.float32)

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()

        meta = dict()
        meta["x"] = torch.from_numpy(self.datas[index]).float()
        return meta


class RotAugCategoryGrasp(Dataset):
    """
    cog centered grasp translation distribution with only positive grasp
    """

    def __init__(self, grasps: List[np.array], rotrepr=None) -> None:
        # grasps List[np.array(N, 7)] xyz xyzw

        self.num_object = len(grasps)
        self.num_grasp = grasps[0].shape[0]
        self.datas = grasps
        self.rotrepr = rotrepr

    def __len__(self):
        return self.num_grasp * self.num_object

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()

        obj_idx = index % self.num_object
        grasp_idx = index // self.num_object

        raw_data = self.datas[obj_idx][grasp_idx]

        if self.rotrepr == "quat":
            quat = uniform_quaternion(1).reshape(-1)
            action_transform = Transform.from_xyzquat(np.hstack([np.zeros(3), quat]))
            data = (action_transform * Transform.from_xyzquat(raw_data)).as_xyzquat()

            condition = np.zeros((self.num_object + 4), dtype=np.float32)
            condition[obj_idx] = 1
            condition[self.num_object :] = quat.astype(np.float32)
        elif self.rotrepr == "rotmat":
            quat = uniform_quaternion(1).reshape(-1)
            action_transform = Transform.from_xyzquat(np.hstack([np.zeros(3), quat]))
            data = (action_transform * Transform.from_xyzquat(raw_data)).as_xyzquat()

            condition = np.zeros((self.num_object + 9), dtype=np.float32)
            condition[obj_idx] = 1
            condition[self.num_object :] = (
                action_transform.rotation.as_matrix().reshape(-1).astype(np.float32)
            )

        elif self.rotrepr == None:
            data = raw_data
            condition = np.zeros((self.num_object), dtype=np.float32)
            condition[obj_idx] = 1

        meta = dict()
        meta["x"] = torch.from_numpy(data).float()
        meta["c"] = torch.from_numpy(condition).float()
        return meta
