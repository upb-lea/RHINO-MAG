import jax
import jax.numpy as jnp
import equinox as eqx


class FrequencySet(eqx.Module):
    """Class to store measurement data for a single material for a single frequency
    but with potentially variable temperatures.

    Args:
        material_name (str): Name of the material as a scalar string.
        frequency (float): Frequency of the measurement as a scalar float.
        H (jax.Array): Magnetic field strength data in shape (n_sequences, sequence_length).
        B (jax.Array): Magnetic flux density data in space (n_sequences, sequence_length).
        T (jax.Array): Temperature data in shape (n_sequences).
    """

    material_name: str
    frequency: float
    H: jax.Array
    B: jax.Array
    T: jax.Array

    @classmethod
    def from_dict(cls, data_dict: dict) -> "FrequencySet":
        """Create a FrequencySet from a dictionary."""
        return cls(
            material_name=data_dict["material_name"],
            frequency=data_dict["frequency"],
            H=jnp.array(data_dict["H"]),
            B=jnp.array(data_dict["B"]),
            T=jnp.array(data_dict["T"]),
        )
