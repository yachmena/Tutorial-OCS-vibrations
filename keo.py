import functools
from abc import ABC
from abc import abstractmethod
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
from numpy.typing import NDArray
from scipy import constants

config.update("jax_enable_x64", True)

G_to_invcm = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * np.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

# Levi-Civita tensor
eps = jnp.array(
    [int((i - j) * (j - k) * (k - i) * 0.5) for i in range(3) for j in range(3) for k in range(3)],
    dtype=jnp.float64,
).reshape(3, 3, 3)

class Molecule(ABC):
    @property
    @abstractmethod
    def masses(self) -> NDArray[np.float_]:
        pass

    @masses.setter
    @abstractmethod
    def masses(self, m: Iterable[float]):
        pass

    @abstractmethod
    def internal_to_cartesian(self, internal_coords):
        pass

    def batch_internal_to_cartesian(self, internal_coords):
        return _batch_coord_transform(self.internal_to_cartesian, internal_coords)

    def pseudo(self, internal_coords):
        return _pseudo(self.internal_to_cartesian, self.masses, internal_coords)

    def Gmat(self, internal_coords):
        return _Gmat(self.internal_to_cartesian, self.masses, internal_coords)

    def Detgmat(self, internal_coords):
        return _Detgmat(self.internal_to_cartesian, self.masses, internal_coords)

    def dDetgmat(self, internal_coords):
        return _dDetgmat(self.internal_to_cartesian, self.masses, internal_coords)

    def dGmat(self, internal_coords):
        return _dGmat(self.internal_to_cartesian, self.masses, internal_coords)


# shifts Cartesian coords to the centre of mass
def com(internal_to_cartesian):
    @functools.wraps(internal_to_cartesian)
    def wrapper_com(self, *args, **kwargs):
        xyz = internal_to_cartesian(self, *args, **kwargs)
        com = jnp.dot(self.masses, xyz) / jnp.sum(self.masses)
        return xyz - com[None, :]

    return wrapper_com

@functools.partial(jax.jit, static_argnums=(0,))
def _gmat(internal_to_cartesian, masses, q):
    xyz_g = jax.jacfwd(internal_to_cartesian)(q)
    tvib = xyz_g
    xyz = internal_to_cartesian(q)
    natoms = xyz.shape[0]
    trot = jnp.transpose(jnp.dot(eps, xyz.T), (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(natoms)])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = jnp.array([jnp.sqrt(masses[i]) for i in range(natoms)])
    tvec = tvec * masses_sq[:, None, None]
    tvec = jnp.reshape(tvec, (natoms * 3, len(q) + 6))
    return jnp.dot(tvec.T, tvec)

# G-big matrix as the inverse of g-small
@functools.partial(jax.jit, static_argnums=(0,))
def _Gmat(internal_to_cartesian, masses, q):
    g = jnp.linalg.inv(_gmat(internal_to_cartesian, masses, q))
    return g * G_to_invcm

@functools.partial(jax.jit, static_argnums=(0,))
def _dGmat(internal_to_cartesian, masses, q):
    def _Gmat_(q):
        return _Gmat(internal_to_cartesian, masses, q)
    return jax.jacfwd(_Gmat_)(q)

@functools.partial(jax.jit, static_argnums=(0,))
def _Detgmat(internal_to_cartesian, masses, q):
    nq = len(q)
    def _gmat_(q):
        return _gmat(internal_to_cartesian, masses, q)
    return jnp.linalg.det(_gmat_(q)[:nq+3, :nq+3])

@functools.partial(jax.jit, static_argnums=(0,))
def _dDetgmat(internal_to_cartesian, masses,q):
    def _Detgmat_(q):
        return _Detgmat(internal_to_cartesian, masses, q)
    return jax.grad(_Detgmat_)(q)

@functools.partial(jax.jit, static_argnums=(0,))
def _hDetgmat(internal_to_cartesian, masses,q):
    def _Detgmat_(q):
        return _Detgmat(internal_to_cartesian, masses, q)
    return jax.jacfwd(jax.jacfwd(_Detgmat_))(q)