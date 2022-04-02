import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
import pyvista as pv
from .geometry import Transform
# =========================================================================
EPS = 1e-7

def l2(u, v):
    return np.sqrt(((u-v)**2).sum())

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

def pv_cannonical_gripper(T, r=1e-3):
    center = pv.Cylinder(
        center=[0, 0, 3.3 * 1e-2], direction=[0, 0, 1], height=6.6 * 1e-2, radius=r
    )
    width = pv.Cylinder(
        center=[0, 0, 6.6 * 1e-2], direction=[1, 0, 0], height=2 * 4.1 * 1e-2, radius=r
    )
    lfinger = pv.Cylinder(
        center=[4.1 * 1e-2, 0, 8.9 * 1e-2],
        direction=[0, 0, 1],
        height=4.6 * 1e-2,
        radius=r,
    )
    rfinger = pv.Cylinder(
        center=[-4.1 * 1e-2, 0, 8.9 * 1e-2],
        direction=[0, 0, 1],
        height=4.6 * 1e-2,
        radius=r,
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
    def __init__(self ,true_grasp:Transform, positive_grasp:Transform, trans_threshold, quat_threshold):
        assert isinstance(true_grasp, Transform)  and isinstance(positive_grasp, Transform), "ERROR"
        
        self.trans_threshold = trans_threshold
        self.quat_threshold  = quat_threshold
        
        true_xyz, true_qaut = true_grasp.as_xyzquat()
        posi_xyz, posi_quat = positive_grasp.as_xyzquat() # N_true x N_positive

        self.trans_cdist = cdist(true_xyz, posi_xyz, "euclidean")
        self.quat_cdist = cdist(true_qaut, posi_quat, "cosine") #1.0 - q1Tq2 
        self.quat_cdist = 2.0 * np.arccos(np.abs(self.quat_cdist - 1.0)) # 2*acros(|q1Tq2|)

        
        #transquat to Tmatrix at grasp mount frame
        to_grasp_mount = Transform.from_xyzquat(np.array([0, 0, -0.112, 0, 0, 0, 1.0]))
        self.true_grasp = true_grasp * to_grasp_mount
        self.positive_grasp = positive_grasp * to_grasp_mount

    ######################  Quantitative Study ###################################
    def get_k_percent_index(self, k):
        assert k>=0 and k<=1, "k in [0, 1]"
        total_samples = self.trans_cdist.shape[1]
        num_samples = int(k * total_samples)

        if num_samples == 0:
            num_samples = 1
        
        index = np.random.choice(np.arange(total_samples), (num_samples, ), replace=False)
        return index, num_samples

    def pr_cov_expcov_k(self, k):
        # precision(higher better): N_true exitst in N_positive / N_positive
        # coverage (higher better) : N_positive exitst in N_true / N_true
        # expcov   (lower better) : exp( - mean( min (cdist))) cov3 in A Billion Ways to Grasp
        dist = 0.1 * self.trans_cdist + self.quat_cdist
        
        index, num_samples = self.get_k_percent_index(k)
        self.trans_cdist[:, index] < self.trans_threshold
        target_mat = (self.trans_cdist[:, index] < self.trans_threshold) * (self.quat_cdist[:, index] < self.quat_threshold)

        return num_samples, np.mean(target_mat.any(0)), np.mean(target_mat.any(1)), np.exp(-np.mean(np.min(dist, 1)))

    ######################## Qualitative  Study ####################################
    def visualize_latent_space(self, prior_z, grasp_z, save_path:str=None):
        '''
        prior_z: sample from pre-defined prior distribution
        grasp_z:
            flow : grasp pose + (condition) -> z transfromation
            vae  : p(z|x, (c)) learned approximate posterior distribution sample
        '''
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

    def visualize_data_space(self, object_mesh: pv.PolyData, object_tex: pv.Texture=None, trans_only:bool=True, save_path:str=None):
        off_screen = True if save_path is not None else False

        T_tg = self.true_grasp.as_matrix()
        T_pg = self.positive_grasp.as_matrix()

        grasp_mesh = pv_cannonical_gripper(np.eye(4))

        plotter = pv.Plotter(shape=(1, 2), window_size=[1280, 1024], off_screen=off_screen)

        plotter.subplot(0, 0)
        plotter.add_text("Real")
        if object_tex is not None:
            plotter.add_mesh(object_mesh, texture=object_tex)
        else:
            plotter.add_mesh(object_mesh, color=[0.7, 0.7, 0.7])

        if trans_only:
            plotter.add_mesh(pv.PolyData(T_tg[:, :3, 3]), color=[0, 1, 0], opacity=0.5)
            
        else:
            for i in range(T_tg.shape[0]):
                plotter.add_mesh(grasp_mesh.transform(T_tg[i], inplace=False), color=[0, 1, 0], opacity=0.2)

        plotter.subplot(0, 1)
        plotter.add_text("Gen")
        
        if object_tex is not None:
            plotter.add_mesh(object_mesh, texture=object_tex)
        else:
            plotter.add_mesh(object_mesh, color=[0.7, 0.7, 0.7])

        if trans_only:
            plotter.add_mesh(pv.PolyData(T_pg[:, :3, 3]), color=[0, 1, 0], opacity=0.5)
        else:
            for i in range(T_pg.shape[0]):
                plotter.add_mesh(grasp_mesh.transform(T_pg[i], inplace=False), color=[0, 1, 0], opacity=0.2)

        plotter.link_views()

        if not off_screen:
            plotter.show(screenshot=True)
            pil_image = Image.fromarray(plotter.image)
            pil_image.save('data_space.png')
        else:
            plotter.screenshot(str(save_path))

    

