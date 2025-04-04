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


class MaterialSet(eqx.Module):
    """Class to store measurement data for a single material but with variable
    frequencies and temperatures.

    Args:
        material_name (str): Name of the material as a scalar string.
        frequency_sets (list[FrequencySet]): List of FrequencySet objects.
        frequencies (jax.Array): Frequencies in Hz as a 1D array.
    """

    material_name: str
    frequency_sets: list[FrequencySet]
    frequencies: jax.Array

    @classmethod
    def from_dict(cls, data_dict: dict) -> "MaterialSet":
        """Create a MaterialSet from a dictionary."""
        return cls(
            material_name=data_dict["material_name"],
            frequency_sets=[FrequencySet.from_dict(fs) for fs in data_dict["frequency_sets"]],
        )

    def __getitem__(self, idx: int) -> FrequencySet:
        """Return the frequency set at the given index."""
        return self.frequency_sets[idx]

    def __iter__(self):
        """Return an iterator over the frequency sets."""
        return iter(self.frequency_sets)

    def get_at_frequency(self, idx: float) -> FrequencySet:
        """Return the frequency set at the given index."""
        return self.frequency_sets[jnp.where(self.frequencies == idx)[0][0]]
