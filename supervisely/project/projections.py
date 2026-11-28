from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from supervisely.api.api import Api


class Projection(ABC):
    """
    This abstract class defines the interface for projection models that
    transform points between 3D camera coordinates and 2D image (lens)
    coordinates. Subclasses must implement all abstract methods.
    """

    @abstractmethod
    def project_3d_to_2d(
        self, cam_points: Union[np.ndarray, list], invalid_value: float = np.nan
    ) -> np.ndarray:
        """
        Project 3D points from the camera frame to 2D image coordinates.

        :param cam_points: 3D points in the camera frame with shape (N, 3).
        :type cam_points: numpy.ndarray
        :param invalid_value: Value to assign for points that cannot be projected.
            Defaults to NaN.
        :type invalid_value: float
        :return: 2D image coordinates for each input point with shape (N, 2).
        :rtype: numpy.ndarray
        """
        ...

    @abstractmethod
    def project_2d_to_3d(
        self, lens_points: Union[np.ndarray, List], norms: Union[np.ndarray, List]
    ) -> np.ndarray:
        """
        Back-project 2D image points into 3D camera space given ray norms.

        :param lens_points: 2D points in the image (lens) plane
        :type lens_points: numpy.ndarray of shape (N, 2)
        :param norms: Distance along each ray to reconstruct the 3D point. May be of shape (N,) or (N, 1)
        :type norms: numpy.ndarray of shape (N,) or (N, 1)
        :returns: 3D points in camera coordinates
        :rtype: numpy.ndarray of shape (N, 3)
        """
        ...

    @staticmethod
    def ensure_point_list(
        points: Union[np.ndarray, List],
        dim: int,
        concatenate: bool = True,
        crop: bool = True,
    ) -> np.ndarray:
        """
        Ensure that the input points are a numpy array of the specified dimension.
        """
        if isinstance(points, list):
            points = np.array(points)
        try:
            assert isinstance(points, np.ndarray)
            assert points.ndim == 2

            if crop:
                for test_dim in range(4, dim, -1):
                    if points.shape[1] == test_dim:
                        new_shape = test_dim - 1
                        assert np.array_equal(
                            points[:, new_shape], np.ones(points.shape[0])
                        )
                        points = points[:, 0:new_shape]

            if concatenate and points.shape[1] == (dim - 1):
                points = np.concatenate(
                    (np.array(points), np.ones((points.shape[0], 1))), axis=1
                )

            if points.shape[1] != dim:
                raise AssertionError(
                    "Points.shape[1] == dim failed ({} != {})".format(
                        points.shape[1], dim
                    )
                )
            return points
        except AssertionError as e:
            if not e.args:
                raise ValueError(
                    "Points must be a numpy array of shape (N, {}) or a list of points with dimension {}. "
                    "Got shape {} instead.".format(dim, dim, points.shape)
                ) from e
            raise ValueError from e


class SphericalProjection(Projection):
    """Projection model for spherical coordinates, where 3D points are projected onto a sphere."""

    def __init__(self, focal_length: Union[float, list]):
        self.focal_length = (
            focal_length if isinstance(focal_length, (float, int)) else focal_length[0]
        )

    @property
    def name(self):
        return "spherical_equidist"

    def project_3d_to_2d(
        self, cam_points: Union[np.ndarray, List], invalid_value=np.nan
    ):
        camera_points = super().ensure_point_list(points=cam_points, dim=3)

        r = np.linalg.norm(camera_points, axis=1)
        theta = np.arctan2(camera_points[:, 1], camera_points[:, 0])
        phi = np.arccos(camera_points[:, 2] / r)

        uv = np.zeros((camera_points.shape[0], 2))
        uv.T[0] = self.focal_length * theta
        uv.T[1] = self.focal_length * phi
        uv[r == 0] = invalid_value
        return uv

    def project_2d_to_3d(
        self, image_points: Union[np.ndarray, List], norms: Union[np.ndarray, List]
    ):
        image_points = super().ensure_point_list(points=image_points, dim=2)
        norms = super().ensure_point_list(points=norms, dim=1)

        theta = image_points[:, 0] / self.focal_length
        phi = image_points[:, 1] / self.focal_length

        outs = np.zeros((image_points.shape[0], 3))
        outs.T[0] = norms * np.sin(phi) * np.cos(theta)
        outs.T[1] = norms * np.sin(phi) * np.sin(theta)
        outs.T[2] = norms * np.cos(phi)
        return outs


class CylindricalProjection(Projection):
    """Projection model for cylindrical coordinates, where 3D points are projected onto a cylinder."""

    def __init__(self, focal_length: Union[float, int, list]):
        self.focal_length = (
            focal_length if isinstance(focal_length, (float, int)) else focal_length[0]
        )

    @property
    def name(self):
        return "cylindrical_equidist"

    def project_3d_to_2d(
        self, cam_points: Union[np.ndarray, List], invalid_value=np.nan
    ):
        camera_points = super().ensure_point_list(cam_points, dim=3)

        theta = np.arctan2(camera_points.T[0], camera_points.T[2])
        chi = np.sqrt(
            camera_points.T[0] * camera_points.T[0]
            + camera_points.T[2] * camera_points.T[2]
        )

        uv = np.zeros((camera_points.shape[0], 2))
        uv.T[0] = self.focal_length * theta
        uv.T[1] = (
            self.focal_length * camera_points.T[1] * np.divide(1, chi, where=(chi != 0))
        )
        uv[chi == 0] = invalid_value
        return uv

    def project_2d_to_3d(
        self, image_points: Union[np.ndarray, List], norms: Union[np.ndarray, List]
    ):
        image_points = super().ensure_point_list(points=image_points, dim=2)
        norms = super().ensure_point_list(points=norms, dim=1)

        outs = np.zeros((image_points.shape[0], 3))

        theta = image_points.T[0] / self.focal_length
        scale = np.divide(
            norms.flat,
            np.sqrt(
                image_points.T[1] * image_points.T[1]
                + self.focal_length * self.focal_length
            ),
        )
        outs.T[0] = self.focal_length * np.sin(theta) * scale
        outs.T[1] = image_points.T[1] * scale
        outs.T[2] = self.focal_length * np.cos(theta) * scale
        return outs


class RadialPolyCamProjection(Projection):
    """Projection model for cameras with radial distortion, using polynomial coefficients."""

    def __init__(self, distortion_params: List[float]):
        self.coefficients = np.asarray(distortion_params)
        self.power = np.array([np.arange(start=1, stop=self.coefficients.size + 1)]).T

    @property
    def name(self):
        return "radial_polycam"

    def project_3d_to_2d(
        self, cam_points: Union[np.ndarray, List], invalid_value: float = np.nan
    ):
        camera_points = super().ensure_point_list(points=cam_points, dim=3)
        chi = np.sqrt(
            camera_points.T[0] * camera_points.T[0]
            + camera_points.T[1] * camera_points.T[1]
        )
        theta = np.pi / 2.0 - np.arctan2(camera_points.T[2], chi)
        rho = self._theta_to_rho(theta)
        lens_points = (
            np.divide(rho, chi, where=(chi != 0))[:, np.newaxis] * camera_points[:, 0:2]
        )

        # set (0, 0, 0) = np.nan
        lens_points[(chi == 0) & (cam_points[:, 2] == 0)] = invalid_value
        return lens_points

    def project_2d_to_3d(
        self, lens_points: Union[np.ndarray, List], norms: Union[np.ndarray, List]
    ):
        lens_points = super().ensure_point_list(points=lens_points, dim=2)
        norms = super().ensure_point_list(points=norms, dim=1).reshape(norms.size)

        rhos = np.linalg.norm(lens_points, axis=1)
        thetas = self._rho_to_theta(rhos)
        chis = norms * np.sin(thetas)
        zs = norms * np.cos(thetas)
        xy = np.divide(chis, rhos, where=(rhos != 0))[:, np.newaxis] * lens_points
        xyz = np.hstack((xy, zs[:, np.newaxis]))
        return xyz

    def _theta_to_rho(self, theta):
        return np.dot(self.coefficients, np.power(np.array([theta]), self.power))

    def _rho_to_theta(self, rho):
        coeff = list(reversed(self.coefficients))
        results = np.zeros_like(rho)
        for i, _r in enumerate(rho):
            theta = np.roots([*coeff, -_r])
            theta = np.real(theta[theta.imag == 0])
            theta = theta[np.where(np.abs(theta) < np.pi)]
            theta = np.min(theta) if theta.size > 0 else 0
            results[i] = theta
        return results


class Camera:
    def __init__(
        self,
        lens: Projection,
        translation: np.ndarray,
        rotation: np.ndarray,
        size: np.ndarray,
        principle_point: np.ndarray,
        aspect_ratio: float = 1.0,
    ):
        self._lens = lens
        self.size = np.array([size[0], size[1]], dtype=int)
        pose = np.eye(4)
        pose[0:3, 3] = translation
        pose[0:3, 0:3] = rotation
        self._pose = np.asarray(pose, dtype=float)
        self._inv_pose = np.linalg.inv(self._pose)
        self._principle_point = (
            0.5 * self.size
            + np.array([principle_point[0], principle_point[1]], dtype=float)
            - 0.5
        )
        self._aspect_ratio = np.array([1, aspect_ratio], dtype=float)

    @property
    def lens(self) -> Projection:
        return self._lens

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def cx(self) -> float:
        return self._principle_point[0]

    @property
    def cy(self) -> float:
        return self._principle_point[1]

    @property
    def cx_offset(self) -> float:
        return self._principle_point[0] - 0.5 * self.size[0] + 0.5

    @property
    def cy_offset(self) -> float:
        return self._principle_point[1] - 0.5 * self.size[1] + 0.5

    @property
    def aspect_ratio(self) -> float:
        return self._aspect_ratio[1]

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self._pose[0:3, 0:3]

    @property
    def translation_vector(self) -> np.ndarray:
        return self._pose[0:3, 3]

    def project_3d_to_2d(
        self, world_points: np.ndarray, do_clip=False, invalid_value=np.nan
    ):
        world_points = Projection.ensure_point_list(points=world_points, dim=4)

        camera_points = world_points @ self._inv_pose.T
        lens_points = self.lens.project_3d_to_2d(
            camera_points[:, 0:3], invalid_value=invalid_value
        )
        screen_points = (lens_points * self._aspect_ratio) + self._principle_point
        return (
            self._apply_clip(screen_points, screen_points) if do_clip else screen_points
        )

    def project_2d_to_3d(
        self, screen_points: np.ndarray, norm: np.ndarray, do_clip=False
    ):
        screen_points = Projection.ensure_point_list(
            points=screen_points, dim=2, concatenate=False, crop=False
        )
        norm = Projection.ensure_point_list(
            points=norm[:, np.newaxis], dim=1, concatenate=False, crop=False
        )
        lens_points = (screen_points - self._principle_point) / self._aspect_ratio
        lens_points = (
            self._apply_clip(lens_points, screen_points) if do_clip else lens_points
        )

        camera_points = self.lens.project_2d_to_3d(lens_points, norm)

        camera_points = Projection.ensure_point_list(points=camera_points, dim=4)
        world_points = camera_points @ self._pose.T
        return world_points[:, 0:3]

    def _apply_clip(self, points, clip_source) -> np.ndarray:
        if self.width == 0 or self.height == 0:
            raise RuntimeError("clipping without a size is not possible")
        mask = (
            (clip_source[:, 0] < 0)
            | (clip_source[:, 0] >= self.width)
            | (clip_source[:, 1] < 0)
            | (clip_source[:, 1] >= self.height)
        )

        points[mask] = [np.nan]
        return points

    def to_json(self):
        if isinstance(self.lens, RadialPolyCamProjection): # <-- no focal length in this model
            raise NotImplementedError("RadialPolyCamProjection not supported yet")
        return {
            "calibration": {
                "intrinsic": {
                    "cx": float(self.cx),
                    "cy": float(self.cy),
                    "fx": float(self.lens.focal_length / 0.005),
                    "fy": float(self.lens.focal_length * self.aspect_ratio / 0.005),
                    "cameraModel": self.lens.name,
                }
            }
        }


def create_img_projection_maps(source_cam: Camera, destination_cam: Camera):
    """generates maps for cv2.remap to remap from one camera to another"""
    import cv2

    u_map = np.zeros(
        (destination_cam.height, destination_cam.width, 1), dtype=np.float32
    )
    v_map = np.zeros(
        (destination_cam.height, destination_cam.width, 1), dtype=np.float32
    )

    destination_points_b = np.arange(destination_cam.height)

    for u_px in range(destination_cam.width):
        destination_points_a = np.ones(destination_cam.height) * u_px
        destination_points = np.vstack((destination_points_a, destination_points_b)).T

        destP = destination_cam.project_2d_to_3d(destination_points, norm=np.array([1]))
        source_points = source_cam.project_3d_to_2d(destP)

        u_map.T[0][u_px] = source_points.T[0]
        v_map.T[0][u_px] = source_points.T[1]

    map1, map2 = cv2.convertMaps(
        u_map, v_map, dstmap1type=cv2.CV_16SC2, nninterpolation=False
    )
    return map1, map2

def add_projection_meta(api: Api, image_id: int, camera: Camera):
    """Add the camera projection info to an image's metadata using API.

    :param api: Supervisely API instance.
    :param image_id: ID of the image to update.
    :param camera: Camera instance with projection information.
    :return: Updated image metadata in JSON format.
    :rtype: dict
    """
    return api.image.update_meta(image_id, camera.to_json())