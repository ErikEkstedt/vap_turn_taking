import torch
import matplotlib.pyplot as plt


def plot_area(oh, ax, label=None, color="b", alpha=1, hatch=None):
    ax.fill_between(
        torch.arange(oh.shape[0]),
        y1=-1,
        y2=1,
        where=oh,
        color=color,
        alpha=alpha,
        label=label,
        hatch=hatch,
    )


def plot_vad_oh(
    vad_oh,
    ax=None,
    colors=["b", "orange"],
    yticks=["B", "A"],
    ylabel=None,
    alpha=1,
    label=(None, None),
    legend_loc="best",
    plot=False,
):
    """
    vad_oh:     torch.Tensor: (N, 2)
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    x = torch.arange(vad_oh.shape[0]) + 0.5  # fill_between step = 'mid'
    ax.fill_between(
        x,
        y1=0,
        y2=vad_oh[:, 0],
        step="mid",
        alpha=alpha,
        color=colors[0],
        label=label[1],
    )
    ax.fill_between(
        x,
        y1=0,
        y2=-vad_oh[:, 1],
        step="mid",
        alpha=alpha,
        label=label[0],
        color=colors[1],
    )
    if label[0] is not None:
        ax.legend(loc=legend_loc)
    ax.hlines(y=0, xmin=0, xmax=len(x), color="k", linestyle="dashed")
    ax.set_xlim([0, vad_oh.shape[0]])
    ax.set_ylim([-1.05, 1.05])

    if yticks is None:
        ax.set_yticks([])
    else:
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(yticks)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_events(
    vad,
    hold=None,
    shift=None,
    event=None,
    event_alpha=0.3,
    vad_alpha=0.6,
    ax=None,
    plot=True,
    figsize=(9, 6),
):
    vad = vad.cpu()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize)

    _ = plot_vad_oh(
        # vad, ax=ax.twinx(), alpha=vad_alpha, legend_loc="upper right", label=["B", "A"]
        vad,
        ax,
        alpha=vad_alpha,
        legend_loc="upper right",
        label=["B", "A"],
    )

    # Twin axis for events
    if hold is not None or shift is not None or event is not None:
        twinax = ax.twinx()

    if hold is not None:
        hold = hold.detach().cpu()
        plot_area(
            hold[:, 0],
            ax=twinax,
            label="Hold -> A",
            color="red",
            alpha=event_alpha,
            hatch="*",
        )
        plot_area(
            hold[:, 1], ax=twinax, label="Hold -> B", color="red", alpha=event_alpha
        )
        twinax.legend(loc="upper left")

    if shift is not None:
        shift = shift.detach().cpu()
        plot_area(
            shift[:, 0],
            ax=twinax,
            label="Shift -> A",
            color="green",
            alpha=event_alpha,
            hatch="*",
        )
        plot_area(
            shift[:, 1], ax=twinax, label="Shift -> B", color="green", alpha=event_alpha
        )
        twinax.legend(loc="upper left")

    if event is not None:
        event = event.detach().cpu()
        plot_area(event, ax=twinax, label="Event", color="purple", alpha=event_alpha)
        twinax.legend(loc="upper left")

    # Twin axis for events
    if hold is not None or shift is not None or event is not None:
        twinax.set_yticks([])
        twinax.set_ylim([-1.05, 1.05])

    ax.set_ylim([-1.05, 1.05])
    ax.set_xlim([0, vad.shape[0]])

    if plot:
        plt.tight_layout()
        plt.pause(0.1)

    return fig, ax
