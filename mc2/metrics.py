from typing import Any
import numpy as np

import jax
import jax.numpy as jnp

from mc2.data_management import MaterialSet
from mc2.models.model_interface import ModelInterface


def get_energy_loss(b: jax.Array, h: jax.Array) -> jax.Array:
    """Compute the energy loss in a hysteresis loop.
    The energy loss is computed as the area of the hysteresis loop.

    Args:
        b (jax.Array): Magnetic flux density in T.
        h (jax.Array): Magnetic field strength in A/m.
    """
    return jnp.trapezoid(h, b)


def worst_case_error(h_est: jax.Array, h_true: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(h_est - h_true))


def mean_absolute_error(h_est: jax.Array, h_true: jax.Array) -> jax.Array:
    return jnp.mean(jnp.abs(h_est - h_true))


def mean_squared_error(h_est: jax.Array, h_true: jax.Array) -> jax.Array:
    return jnp.mean(jnp.abs(h_est - h_true) ** 2)


def default_metrics():
    bh_metrics = {"energy_loss": get_energy_loss}
    h_metrics = {
        "WCE": worst_case_error,
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
    }

    return {"bh": bh_metrics, "h": h_metrics}


def evaluate_model_estimation(B: jax.Array, H: jax.Array, H_est: jax.Array, metrics: dict) -> dict:

    performance_metric_values = {"bh": {}, "h": {}}

    for name, function in metrics["bh"].items():
        performance_metric_values["bh"][name] = function(B, H_est)
    for name, function in metrics["h"].items():
        performance_metric_values["h"][name] = function(H_est, H)
    return performance_metric_values


def evaluate_model(
    model: ModelInterface,
    B_past: jax.Array,
    H_past: jax.Array,
    B_future: jax.Array,
    H_future: jax.Array,
    T: float,
    metrics: dict = None,
    reduce_to_scalar: bool = True,
) -> dict[str, Any]:
    if metrics is None:
        metrics = default_metrics()

    H_est = model(B_past, H_past, B_future, T)

    metric_results = jax.vmap(evaluate_model_estimation, in_axes=(0, 0, 0, None))(B_future, H_future, H_est, metrics)

    if reduce_to_scalar:
        for name, result in metric_results["bh"].items():
            metric_results["bh"][name] = jnp.mean(result).item()
        for name, result in metric_results["h"].items():
            metric_results["h"][name] = jnp.mean(result).item()

    return metric_results


def evaluate_model_on_test_set(
    model: ModelInterface,
    test_set: MaterialSet,
    metrics: dict = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    eval_metrics = {}
    for frequency in test_set.frequencies:
        test_set_at_frequency = test_set.at_frequency(frequency)
        eval_metrics[frequency.item()] = {
            sequence_length.item(): evaluate_model(
                model,
                B_past=test_set_at_frequency.B[:, :20],
                H_past=test_set_at_frequency.H[:, :20],
                B_future=test_set_at_frequency.B[:, 20 : 20 + sequence_length],
                H_future=test_set_at_frequency.H[:, 20 : 20 + sequence_length],
                T=test_set_at_frequency.T[:],
                metrics=metrics,
                reduce_to_scalar=True,
            )
            for sequence_length in np.linspace(10, test_set_at_frequency.H.shape[-1], 10, dtype=int)
        }
    return eval_metrics
