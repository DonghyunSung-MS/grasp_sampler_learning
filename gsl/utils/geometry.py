from typing import Tuple

import numpy as np
import roma
import torch
from scipy.spatial.transform import Rotation


def torch_transform2transrotvec(T: torch.Tensor):
    trans = T[..., :3, 3]
    rot_mat = T[..., :3, :3]
    rotvec = roma.rotmat_to_rotvec(rot_mat)
    out = torch.cat([trans, rotvec], -1)
    # print(out)
    return out


def torch_transform2transquat(T: torch.Tensor):
    trans = T[..., :3, 3]
    rot_mat = T[..., :3, :3]
    quat = roma.rotmat_to_unitquat(rot_mat)
    return torch.cat([trans, quat], -1)


def torch_transquat2transrotvec(xyzquat: torch.Tensor):
    trans = xyzquat[:, :3]
    rotvec = roma.unitquat_to_rotvec(xyzquat[:, 3:])
    return torch.cat([trans, rotvec], -1)


# http://planning.cs.uiuc.edu/node198.html
def uniform_quaternion(N):
    u1, u2, u3 = np.random.uniform(0, 1, size=(3, N))

    return np.vstack(
        [
            np.sqrt(1 - u1) * np.sin(np.pi * 2 * u2),
            np.sqrt(1 - u1) * np.cos(np.pi * 2 * u2),
            np.sqrt(u1) * np.sin(np.pi * 2 * u3),
            np.sqrt(u1) * np.cos(np.pi * 2 * u3),
        ]
    ).T


class Transform(object):
    def __init__(self, rotation, translation):
        assert isinstance(rotation, Rotation)
        assert isinstance(translation, np.ndarray)

        self.rotation = rotation
        self.translation = translation

        if self.translation.ndim == 1:
            self.num = None
        else:
            self.num = self.translation.shape[0]

    def as_matrix(self) -> np.array:
        if self.num is not None:
            tmp = np.zeros((self.num, 4, 4))
            idx = np.arange(4)
            tmp[:, idx, idx] = 1.0
        else:
            tmp = np.eye(4)

        tmp[..., :3, :3] = self.rotation.as_matrix()
        tmp[..., :3, 3] = self.translation
        return tmp

    def as_xyzquat(self) -> Tuple:
        return self.translation, self.rotation.as_quat()

    def as_xyzrotvec(self) -> Tuple:
        return np.hstack([self.translation, self.rotation.as_rotvec()])

    ############### class method
    @classmethod
    def from_xyzrotvec(cls, xyzrotvec: np.array):
        rotation = Rotation.from_rotvec(xyzrotvec[..., 3:])
        translation = xyzrotvec[..., :3]
        return cls(rotation, translation)

    @classmethod
    def from_euler(cls, xyz, euler, order: str = "xyz", degrees: bool = False):
        """
        translation only from unit(m) euler
        euler: degree or radian with degrees option
        """
        rotation = Rotation.from_euler(order, euler, degrees)
        translation = xyz
        return cls(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[..., :3, :3])
        translation = m[..., :3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_xyzquat(cls, xyzquat):
        rotation = Rotation.from_quat(xyzquat[..., 3:])
        translation = xyzquat[..., :3]
        return cls(rotation, translation)

    @classmethod
    def identity(cls, num_samples=None):
        """Initialize with the identity transformation."""
        rotation = Rotation.identity(num_samples)
        if num_samples is None:
            translation = np.zeros((3))
        else:
            translation = np.zeros((num_samples, 3))
        return cls(rotation, translation)

    ############### operator
    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def inv(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)
