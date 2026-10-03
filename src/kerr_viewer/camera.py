"""Euclidean navigation camera, with no relativistic observer semantics."""

from dataclasses import dataclass, field
from math import atan2, cos, pi, radians, sin, tan

import numpy as np


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


@dataclass(slots=True)
class FreeCamera:
    position: np.ndarray = field(default_factory=lambda: np.array([8., -12., 6.]))
    yaw: float = 0.
    pitch: float = 0.
    speed: float = 3.

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64, copy=True)
        if self.position.shape != (3,) or not np.all(np.isfinite(self.position)):
            raise ValueError("camera position must be a finite 3D vector")
        self.look_at(np.zeros(3))

    def forward(self):
        return np.array([cos(self.pitch)*cos(self.yaw),
                         cos(self.pitch)*sin(self.yaw), sin(self.pitch)])

    def right(self):
        return _unit(np.cross(self.forward(), np.array([0., 0., 1.])))

    def up(self):
        return np.cross(self.right(), self.forward())

    def look_at(self, target):
        direction = _unit(np.asarray(target, dtype=np.float64) - self.position)
        self.yaw = atan2(direction[1], direction[0])
        self.pitch = atan2(direction[2], np.linalg.norm(direction[:2]))

    def turn(self, dx, dy, sensitivity=0.003):
        self.yaw += float(dx)*sensitivity
        self.pitch = float(np.clip(self.pitch - float(dy)*sensitivity,
                                   -pi/2 + 0.002, pi/2 - 0.002))

    def orbit(self, dx, dy, sensitivity=0.003):
        distance = float(np.linalg.norm(self.position))
        if distance < 1e-6:
            return
        self.turn(dx, dy, sensitivity)
        self.position = -distance*self.forward()

    def move(self, forward=0., lateral=0., vertical=0., dt=0.):
        direction = (float(forward)*self.forward() + float(lateral)*self.right()
                     + float(vertical)*np.array([0., 0., 1.]))
        if np.any(direction):
            self.position += _unit(direction)*self.speed*max(0., float(dt))

    def view(self):
        basis = np.stack((self.right(), self.up(), -self.forward()))
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = basis
        matrix[:3, 3] = -basis @ self.position
        return matrix

    @staticmethod
    def projection(aspect, fov_degrees=55., near=0.025, far=90.):
        if aspect <= 0 or near <= 0 or far <= near:
            raise ValueError("invalid perspective parameters")
        f = 1./tan(radians(fov_degrees)/2.)
        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[0, 0] = f/aspect
        matrix[1, 1] = f
        matrix[2, 2] = -(far+near)/(far-near)
        matrix[2, 3] = -2*far*near/(far-near)
        matrix[3, 2] = -1.
        return matrix
