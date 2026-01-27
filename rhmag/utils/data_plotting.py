import matplotlib.pyplot as plt
import jax.numpy as jnp

from rhmag.utils.model_evaluation import get_mixed_frequency_arrays

def plot_single_sequence(B, H, T, t=None, fig=None, axs=None):
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    fig.suptitle("Temperature: " + str(T) + " C°")
    if t is None:
        axs[0].plot(B)
        axs[1].plot(H)

    else:
        axs[0].plot(t, B)
        axs[1].plot(t, H)
        for ax in axs:
            ax.set_xlabel("Time in s")

    axs[0].set_ylabel("B in T")
    axs[1].set_ylabel("H in A/m")

    for ax in axs:
        ax.grid()

    fig.tight_layout()
    return fig, axs


def plot_hysteresis(B, H, T, fig=None, axs=None):

    if fig is None or axs is None:
        fig, axs = plt.subplots(figsize=(10, 10))

    fig.suptitle("Temperature: " + str(T) + " C°")
    axs.plot(H, B)
    axs.set_xlabel("H in A/m")
    axs.set_ylabel("B in T")
    axs.grid()
    fig.tight_layout()
    return fig, axs


def plot_sequence_prediction(B, H, T, H_pred, past_size, figsize=(6, 6)):
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    length = H.shape[-1]
    k = jnp.linspace(0, length - 1, length)

    axs[0].plot(k, B)
    axs[1].plot(k, H, label="H_true")
    axs[1].plot(k[past_size:], H_pred, linestyle="--", color="tab:orange", label="H_pred")

    axs[0].set_ylabel("B in T")
    axs[1].set_ylabel("H in A/m")
    axs[-1].set_xlabel("k")

    for ax in axs:
        ax.grid(alpha=0.3)
    axs[-1].legend()

    fig.tight_layout()
    return fig, axs


def plot_hysteresis_prediction(B, H, T, H_pred, past_size):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig, axs = plot_hysteresis(B, H, T, fig, axs)
    axs.plot(H_pred, B[past_size:])

    return fig, axs


def plot_frequency_sweep(material_set, loader_key, figsize=(30, 8), sequence_length=1000, batch_size=1):
    # gather data

    H, B, T = get_mixed_frequency_arrays(
        material_set,
        sequence_length=sequence_length,
        batch_size=batch_size,
        key=loader_key,
    )

    if batch_size == 1:

        # plot
        fig, axs = plt.subplots(3, 7, figsize=figsize)
        for freq_idx in range(len(material_set.frequencies)):
            axs[0, freq_idx].plot(B[freq_idx])
            axs[1, freq_idx].plot(H[freq_idx])

            axs[2, freq_idx].plot(H[freq_idx], B[freq_idx])

            axs[0, freq_idx].grid(True, alpha=0.3)
            axs[1, freq_idx].grid(True, alpha=0.3)
            axs[2, freq_idx].grid(True, alpha=0.3)

            axs[0, freq_idx].set_ylabel("B")
            axs[0, freq_idx].set_xlabel("k")
            axs[1, freq_idx].set_ylabel("H")
            axs[1, freq_idx].set_xlabel("k")
            axs[2, freq_idx].set_ylabel("B")
            axs[2, freq_idx].set_xlabel("H")

            axs[0, freq_idx].set_title("frequency: " + str(int(material_set.frequencies[freq_idx] / 1e3)) + " kHz")

        fig.tight_layout()
        return fig, axs

    else:
        for batch_idx in range(batch_size):
            fig, axs = plt.subplots(3, 7, figsize=figsize)
            for freq_idx in range(len(material_set.frequencies)):
                axs[0, freq_idx].plot(B[freq_idx, batch_idx])
                axs[1, freq_idx].plot(H[freq_idx, batch_idx])

                axs[2, freq_idx].plot(H[freq_idx, batch_idx], B[freq_idx, batch_idx])

                axs[0, freq_idx].grid(True, alpha=0.3)
                axs[1, freq_idx].grid(True, alpha=0.3)
                axs[2, freq_idx].grid(True, alpha=0.3)

                axs[0, freq_idx].set_ylabel("B")
                axs[0, freq_idx].set_xlabel("k")
                axs[1, freq_idx].set_ylabel("H")
                axs[1, freq_idx].set_xlabel("k")
                axs[2, freq_idx].set_ylabel("B")
                axs[2, freq_idx].set_xlabel("H")

                axs[0, freq_idx].set_title("frequency: " + str(int(material_set.frequencies[freq_idx] / 1e3)) + " kHz")

            fig.tight_layout()
            plt.show()