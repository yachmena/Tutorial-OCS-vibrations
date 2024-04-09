from molecule import masses
import functools
from jax import numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


def com(internal_to_cartesian):
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
