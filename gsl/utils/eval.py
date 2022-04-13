from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from PIL import Image
from scipy.spatial.distance import cdist

from .geometry import Transform, uniform_quaternion

# =========================================================================
EPS = 1e-7


def l2(u, v):
    return np.sqrt(((u - v) ** 2).sum())


def quat_metric(q1, q2):
    # arccros(|<q1, q2>|)
    return np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1.0 + EPS, 1.0 - EPS))


def trans_quat_metric(a, b):
    alpha = 0.1
    t1 = a[:3]
    t2 = b[:3]

    q1 = a[3:]
    q2 = b[3:]

    return alpha * l2(t1, t2) + quat_metric(q1, q2)


def pv_cannonical_gripper(T, r=1e-2):
    # center = pv.Cylinder(
    #     center=[0, 0, 3.3 * 1e-2], direction=[0, 0, 1], height=6.6 * 1e-2, radius=r
    # )
    # width = pv.Cylinder(
    #     center=[0, 0, 6.6 * 1e-2], direction=[1, 0, 0], height=2 * 4.1 * 1e-2, radius=r
    # )
    # lfinger = pv.Cylinder(
    #     center=[4.1 * 1e-2, 0, 8.9 * 1e-2],
    #     direction=[0, 0, 1],
    #     height=4.6 * 1e-2,
    #     radius=r,
    # )
    # rfinger = pv.Cylinder(
    #     center=[-4.1 * 1e-2, 0, 8.9 * 1e-2],
    #     direction=[0, 0, 1],
    #     height=4.6 * 1e-2,
    #     radius=r,
    # )
    center = pv.Cube(
        center=[0, 0, 3.3 * 1e-2], x_length=r, y_length=r, z_length=6.6 * 1e-2
    )
    width = pv.Cube(
        center=[0, 0, 6.6 * 1e-2], x_length=2 * 4.1 * 1e-2, y_length=r, z_length=r
    )
    lfinger = pv.Cube(
        center=[4.1 * 1e-2 - r / 2.0, 0, 8.9 * 1e-2],
        x_length=r,
        y_length=r,
        z_length=4.6 * 1e-2,
    )
    rfinger = pv.Cube(
        center=[-4.1 * 1e-2 + r / 2.0, 0, 8.9 * 1e-2],
        x_length=r,
        y_length=r,
        z_length=4.6 * 1e-2,
    )
    # gripper = pv.MultiBlock()
    # gripper.append(center.transform(T))
    # gripper.append(width.transform(T))
    # gripper.append(lfinger.transform(T))
    # gripper.append(rfinger.transform(T))
    gripper = pv.PolyData()
    gripper.merge(center.transform(T), inplace=True)
    gripper.merge(width.transform(T), inplace=True)
    gripper.merge(lfinger.transform(T), inplace=True)
    gripper.merge(rfinger.transform(T), inplace=True)

    return gripper


# evaluation at grasp center and rotation
class GraspEvaluation:
    def __init__(
        self,
        true_grasp: Transform,
        positive_grasp: Transform,
        trans_threshold,
        quat_threshold,
    ):
        assert isinstance(true_grasp, Transform) and isinstance(
            positive_grasp, Transform
        ), "ERROR"

        self.trans_threshold = trans_threshold
        self.quat_threshold = quat_threshold
        self.w = 1000.0 * np.pi / 180.0 * 15  # 15mm == 1 deg

        true_xyz, true_qaut = true_grasp.as_xyzquat()
        posi_xyz, posi_quat = positive_grasp.as_xyzquat()  # N_true x N_positive

        # posi_quat = uniform_quaternion(posi_quat.shape[0])

        self.trans_cdist = cdist(true_xyz, posi_xyz, "euclidean")
        self.quat_cdist = cdist(true_qaut, posi_quat, "cosine")  # 1.0 - q1Tq2
        self.quat_cdist = 2.0 * np.arccos(
            np.abs(self.quat_cdist - 1.0)
        )  # 2*acros(|q1Tq2|)

        self.dist = self.w * self.trans_cdist + self.quat_cdist
        self.target_mat = (
            self.dist < self.w * self.trans_threshold + self.quat_threshold
        )

        self.true_grasp = true_grasp
        self.positive_grasp = positive_grasp

        # transquat to Tmatrix at grasp mount frame
        # to_grasp_mount = Transform.from_xyzquat(np.array([0, 0, -0.112, 0, 0, 0, 1.0]))
        # self.true_grasp = true_grasp * to_grasp_mount
        # self.positive_grasp = positive_grasp * to_grasp_mount

    ######################  Quantitative Study ###################################
    def get_k_percent_index(self, k):
        assert k >= 0 and k <= 1, "k in [0, 1]"
        total_samples = self.trans_cdist.shape[1]
        num_samples = int(k * total_samples)

        if num_samples == 0:
            num_samples = 1

        index = np.random.choice(
            np.arange(total_samples), (num_samples,), replace=False
        )
        return index, num_samples

    def pr_cov_asp_k(self, k):
        # precision(higher better): N_true exitst in N_positive / N_positive
        # coverage (higher better) : N_positive exitst in N_true / N_true
        # average shortes path (lower better) : mean( min (cdist)) cov3 in A Billion Ways to Grasp

        index, num_samples = self.get_k_percent_index(k)

        return (
            num_samples,
            np.mean(self.target_mat[:, index].any(0)),
            np.mean(self.target_mat[:, index].any(1)),
            np.mean(np.min(self.dist[:, index], 1)),
        )

    # def batch_pr_cov_asp_k(self, ks:np.array):
    #     assert (ks>=0).all() and (ks<=1).all()
    #     dist = self.w * self.trans_cdist + self.quat_cdist
    #     total_samples = dist.shape[0]

    #     num_samples_batch = (total_samples * ks).astype(np.int64)
    #     prs = []
    #     covs = []
    #     asps = []
    #     for

    ######################## Qualitative  Study ####################################
    def visualize_latent_space(self, prior_z, grasp_z, save_path: str = None):
        """
        prior_z: sample from pre-defined prior distribution
        grasp_z:
            flow : grasp pose + (condition) -> z transfromation
            vae  : p(z|x, (c)) learned approximate posterior distribution sample
        """
        df = pd.DataFrame()
        num_data = prior_z.shape[0]
        for i in range(prior_z.shape[1] + 1):
            if i == 0:
                df["class"] = ["prior"] * num_data + ["grasp"] * num_data
            else:
                df[f"dim{i}"] = np.hstack([prior_z[:, i - 1], grasp_z[:, i - 1]])

        g = sns.PairGrid(df, hue="class")
        # g.map_upper(sns.kdeplot)
        g.map_upper(sns.scatterplot, alpha=0.2)

        # g.map_lower(sns.kdeplot)
        g.map_lower(sns.scatterplot, alpha=0.2)

        g.map_diag(sns.histplot, kde=True)

        g.add_legend()

        if save_path is None:
            plt.savefig("latent_space.png", dpi=300)
            plt.show()
        else:
            plt.savefig(f"{save_path}", dpi=300)

    def visualize_data_space(
        self,
        object_mesh: pv.PolyData,
        object_tex: pv.Texture = None,
        trans_only: bool = True,
        save_dir: Path = None,
    ):
        off_screen = True if save_dir is not None else False

        T_tg = self.true_grasp.as_matrix()
        T_pg = self.positive_grasp.as_matrix()
        if not trans_only:
            index = np.random.randint(0, T_tg.shape[0], 20)
            T_pg = T_pg[index]
            T_tg = T_tg[index]

        grasp_mesh = pv_cannonical_gripper(np.eye(4))
        width = 640
        height = 480
        plotter = pv.Plotter(
            shape=(1, 2), window_size=[width * 2, height], off_screen=off_screen
        )

        cpos = (0.5, 0.5, 0.5)
        plotter.camera_position = [cpos, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

        plotter.set_background("white")
        plotter.subplot(0, 0)
        plotter.add_text("Real")
        if object_tex is not None:
            plotter.add_mesh(object_mesh, texture=object_tex)
        else:
            plotter.add_mesh(object_mesh, color=[0.7, 0.7, 0.7])

        if trans_only:
            plotter.add_mesh(pv.PolyData(T_tg[:, :3, 3]), color=[0, 1, 0], opacity=0.7)

        else:
            for i in range(T_tg.shape[0]):
                plotter.add_mesh(
                    grasp_mesh.transform(T_tg[i], inplace=False),
                    color=[0, 1, 0],
                    opacity=0.5,
                )

        plotter.subplot(0, 1)
        plotter.add_text("Gen")

        if object_tex is not None:
            plotter.add_mesh(object_mesh, texture=object_tex)
        else:
            plotter.add_mesh(object_mesh, color=[0.7, 0.7, 0.7])

        if trans_only:
            plotter.add_mesh(pv.PolyData(T_pg[:, :3, 3]), color=[0, 1, 0], opacity=0.7)
        else:
            for i in range(T_pg.shape[0]):
                plotter.add_mesh(
                    grasp_mesh.transform(T_pg[i], inplace=False),
                    color=[0, 1, 0],
                    opacity=0.5,
                )

        plotter.link_views()
        # plotter.add_axes_at_origin()

        if not off_screen:
            plotter.show(screenshot=True)
            pil_image = Image.fromarray(plotter.image)
            pil_image.save("data_space.png")
        else:
            name = "trans" if trans_only else "grasp"
            img = plotter.screenshot()
            real_image = Image.fromarray(img[:, :width])
            gen_image = Image.fromarray(img[:, width:])
            real_image.save(save_dir / f"real_{name}.png")
            gen_image.save(save_dir / f"gen_{name}.png")
