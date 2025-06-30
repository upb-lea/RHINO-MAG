import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import AVAILABLE_MATERIALS, MaterialSet, FrequencySet


SUPPORTED_ARCHS = ["NODE"]
