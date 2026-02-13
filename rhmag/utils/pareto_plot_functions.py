import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np


def color_helper(model_type, colors, color_others):
    for key, color in colors.items():
        if key in model_type:
            return color
    return color_others


def visualize_pareto_cross_model(
    df,
    df_external,
    metrics,
    colors,
    color_others,
    sharex,
    sharey,
    xlim,
    show_median,
    scale_log_metric=True,
    scale_log_size=True,
):
    df = df.copy()
    df_external = df_external.copy()

    df["color"] = df["model_type"].apply(lambda x: color_helper(x, colors, color_others))

    df_external["color"] = color_others

    fig, axs = plt.subplots(
        nrows=1, ncols=len(metrics), sharex=sharex, sharey=sharey, figsize=(7.167, 7.167 / 2), squeeze=False
    )

    for i, metric in enumerate(metrics):
        ax = axs[0, i]
        target_col = f"{metric}_95th"

        # Start own results
        for color, group_df in df.groupby("color"):
            current_color = color
            sns.scatterplot(
                data=group_df,
                x=target_col,
                y="n_params",
                color=current_color,
                marker="o",
                s=20,
                alpha=0.8 if not show_median else 0.1,
                ax=ax,
                zorder=3,
            )

            if show_median:
                med_df = group_df.groupby("model_type").median(numeric_only=True)
                sorted_indices = np.argsort(med_df.n_params)
                med_df = med_df.iloc[sorted_indices]

                median = med_df[f"{metric}_95th"]
                ax.plot(median, med_df["n_params"], c=current_color, alpha=0.8)
        # End own results

        # Start external results:
        sns.scatterplot(
            data=df_external,
            x=target_col,
            y="n_params",
            color=color_others,
            marker="s",
            s=20,
            alpha=0.8,
            ax=ax,
            zorder=3,
        )
        for _, row in df_external.iterrows():
            ax.text(
                row[target_col] * 1.05,
                row["n_params"],
                str(row["model_type"]),
                fontsize=10,
                alpha=1,
                va="center",
                color=color_others,
                fontweight="normal",
            )
        # End external results

        if scale_log_metric:
            ax.set_xscale("log")
        if scale_log_size:
            ax.set_yscale("log")

        unique_params = sorted(df["n_params"].unique().astype(int))
        ax.set_yticks(unique_params)
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())

        ax.set_yticklabels(unique_params, rotation=0, fontsize=9)  # ha='right')
        ax.tick_params(which="major", axis="y", direction="in")
        ax.tick_params(which="both", axis="x", direction="in")
        ax.yaxis.minorticks_off()
        ax.xaxis.minorticks_off()

        if i == 0:
            # ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6], minor=True)
            # ax.set_xticks([0.1], minor=False)

            ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
            if xlim is not None:
                # ax.set_xlim(0.08[0][0], 0.75)
                ax.set_xlim(xlim[0][0], xlim[0][1])

        if i == 1:
            # ax.set_xticks([0.02, 0.05, 0.2], minor=True)
            # ax.set_xticks([0.1], minor=False)

            ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
            if xlim is not None:
                # ax.set_xlim(0.013, 0.22)
                ax.set_xlim(xlim[1][0], xlim[1][1])

        if i == 0:
            ax.set_ylabel("\# Model params.", fontsize=9)

        label_map = {"sre": "95-th percentile SRE", "nere": "95-th percentile NERE"}
        ax.set_xlabel(label_map.get(metric, metric), fontsize=9)

        ax.grid(True, which="both", ls="--", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color=color, label=model_label, markersize=5) for model_label, color in colors.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor=color_others, label="External", markersize=5)
    )

    fig.legend(handles=legend_elements)
    fig.tight_layout()
    return fig, axs
