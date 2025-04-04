import matplotlib.pyplot as plt


def plot_single_sequence(B, H, T, t=None, fig=None, axs=None):
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    fig.suptitle("Temperature: " + str(T) + " C°")
    if t is None:
        axs[0].plot(B)
        axs[1].plot(H)

        for ax in axs:
            ax.set_xlabel("Time in s")

    else:
        axs[0].plot(t, B)
        axs[1].plot(t, H)

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
