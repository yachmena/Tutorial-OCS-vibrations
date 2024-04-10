"""Computes molecular rotational-vibrational kinetic-energy operator"""

from molecule import masses
import functools
import jax
from jax import numpy as jnp
from jax import config
from scipy import constants
import numpy as np
from typing import Callable

config.update("jax_enable_x64", True)


# multiply KEO G-matrix with `G_to_invcm` to obtain units cm^{-1},
# providing Cartesian coordinates of atoms are in Angstrom
G_to_invcm = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * np.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

# Levi-Civita tensor
eps = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)


def com(internal_to_cartesian: Callable[[np.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Shifts Cartesian coordinates of atoms, provided by the `internal_to_cartesian`
    function to the centre of mass. The atomic masses are accessed from `molecule.masses`.
    Note, the order of atoms in the output of `internal_to_cartesian` must
    match the order of atoms in `molecule.masses`.
    """

    @functools.wraps(internal_to_cartesian)
    def wrapper(*args, **kwargs):
        xyz = internal_to_cartesian(*args, **kwargs)
        com = jnp.dot(jnp.array(masses), xyz) / jnp.sum(jnp.array(masses))
        return xyz - com[None, :]

    return wrapper


def gmat(
    vibrational_coords: np.ndarray,
    internal_to_cartesian: Callable[[np.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Computes the small-g matrix.

    Args:
        vibrational_coords (np.ndarray): An array of 3N-6 vibrational coordinates,
            where N is the number of atoms in the molecule.
        internal_to_cartesian (Callable[[np.ndarray], jnp.ndarray]): A function that transforms
            the vibrational coordinates into Cartesian coordinates.
            This function should accept vibrational coordinates in `vibrational_coords`
            as input and return an array of Cartesian coordinates.

    Returns:
        (jnp.ndarray): The small-g matrix.
    """
    xyz_g = jax.jacfwd(internal_to_cartesian)(vibrational_coords)
    tvib = xyz_g
    xyz = internal_to_cartesian(vibrational_coords)
    natoms = xyz.shape[0]
    trot = jnp.transpose(jnp.dot(eps, xyz.T), (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(natoms)])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = np.array([np.sqrt(masses[i]) for i in range(natoms)])
    tvec = tvec * masses_sq[:, jnp.newaxis, jnp.newaxis]
    tvec = jnp.reshape(tvec, (natoms * 3, len(vibrational_coords) + 6))
    return jnp.dot(tvec.T, tvec)


def Gmat(
    vibrational_coords: np.ndarray,
    internal_to_cartesian: Callable[[np.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Computes the big-G matrix.

    Args:
        vibrational_coords (np.ndarray): An array of 3N-6 vibrational coordinates,
            where N is the number of atoms in the molecule.
        internal_to_cartesian (Callable[[np.ndarray], jnp.ndarray]): A function that transforms
            the vibrational coordinates into Cartesian coordinates.
            This function should accept vibrational coordinates in `vibrational_coords`
            as input and return an array of Cartesian coordinates.

    Returns:
        (jnp.ndarray): The big-G matrix.
    """
    return jnp.linalg.inv(gmat(vibrational_coords, internal_to_cartesian)) * G_to_invcm


# vectorized version of `Gmat`
Gmat_batch = jax.jit(jax.vmap(Gmat, in_axes=(0, None)), static_argnums=(1,))
