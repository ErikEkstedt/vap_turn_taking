import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle


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


def plot_projection_window(
    proj_win,
    bin_frames=None,
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
    proj_win:     torch.Tensor: (2, N)
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Create X from bin_frames/oh-bins
    if bin_frames is None:
        x = torch.arange(proj_win.shape[1])  # 0, 1, 2, 3
        xmax = len(x)
        bin_lines = x[1:].tolist()
        # add last end point n+1
        # Using 'step' in plot and 'post' so we must add values to the right to get correct fill
        # x: 0, 1, 2, 3, 4
        x = torch.cat((x, x[-1:] + 1))
    else:
        assert (
            len(bin_frames) == proj_win.shape[1]
        ), "len(bin_frames) != proj_win.shape[0]"
        if isinstance(bin_frames, torch.Tensor):
            x = bin_frames
        else:
            x = torch.tensor(bin_frames)
        # Add first 0 start point
        # [0, 20, 40, 60, 80]
        x = torch.cat((torch.zeros((1,)), x))
        x = x.cumsum(0)
        xmax = x[-1]
        bin_lines = x[:-1].long().tolist()

    # add last to match x (which includes 0)
    proj_win = torch.cat((proj_win, proj_win[:, -1:]), dim=-1)

    # Fill bins with color
    ax.fill_between(
        x,
        y1=0,
        y2=proj_win[0],
        step="post",
        alpha=alpha,
        color=colors[0],
        label=label[1],
    )
    ax.fill_between(
        x,
        y1=0,
        y2=-proj_win[1],
        step="post",
        alpha=alpha,
        label=label[0],
        color=colors[1],
    )
    # Add lines separating bins
    ax.hlines(y=0, xmin=0, xmax=xmax, color="k", linestyle="dashed")
    for step in bin_lines:
        ax.vlines(x=step, ymin=-1, ymax=1, color="k")
    if label[0] is not None:
        ax.legend(loc=legend_loc)

    # set X/Y-ticks/labels
    ax.set_xlim([0, xmax])
    ax.set_xticks(x[:-1])
    ax.set_xticklabels(x[:-1].tolist())
    ax.set_ylim([-1.05, 1.05])
    if yticks is None:
        ax.set_yticks([])
    else:
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(yticks)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if plot:
        plt.tight_layout()
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


#########################################################################
def plot_backchannel_prediction(vad, bc_pred, bc_color="r", linewidth=2, plot=False):
    n_frames = bc_pred.shape[1]
    if bc_pred.ndim == 3:
        B = bc_pred.shape[0]
        fig, ax = plt.subplots(B, 1, sharex=True, sharey=True)
        for b in range(B):
            plot_vad_oh(vad[b, :n_frames], ax=ax[b])
            ax[b].plot(bc_pred[b, :, 0], color=bc_color, linewidth=linewidth)
            ax[b].plot(-bc_pred[b, :, 1], color=bc_color, linewidth=linewidth)
    else:
        assert vad.ndim == 2, "no batch dimension VAD must be (N, 2)"
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        plot_vad_oh(vad[:n_frames], ax=ax)
        ax.plot(bc_pred[:, 0], color=bc_color, linewidth=linewidth)
        ax.plot(-bc_pred[:, 1], color=bc_color, linewidth=linewidth)

    if plot:
        plt.pause(0.01)

    return fig, ax


#########################################################################
# Plot multiple projection windows
#########################################################################
def plot_all_projection_windows(
    proj_wins, bin_times=[0.2, 0.4, 0.6, 0.8], vad_hz=100, figsize=(12, 9), plot=True
):
    bin_frames = [b * vad_hz for b in bin_times]
    n = proj_wins.shape[0]
    n_cols = 4
    n_rows = math.ceil(n / n_cols)
    # figsize = (4 * n_cols, 3 * n_rows)
    print("n: ", n)
    print("(cols,rows): ", n_cols, n_rows)
    print("figsize: ", figsize)
    if n_rows == 1:
        fig, ax = plt.subplots(1, n_cols, sharex=True, sharey=True, figsize=figsize)
        for col in range(n_cols):
            if col >= n:
                break
            _ = plot_projection_window(
                proj_wins[col], bin_frames=bin_frames, ax=ax[col]
            )
        ax[0].set_xticks([])
        ax[0].set_yticks([])
    else:
        i = 0
        fig, ax = plt.subplots(
            n_rows, n_cols, sharex=True, sharey=True, figsize=figsize
        )
        for row in range(n_rows):
            for col in range(n_cols):
                _ = plot_projection_window(
                    proj_wins[i], bin_frames=bin_frames, ax=ax[row, col]
                )
                i += 1
                if i >= n:
                    break
            if i >= n:
                break

        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
    plt.tight_layout()
    if plot:
        plt.pause(0.1)

    return fig, ax


#########################################################################
# Template
#########################################################################
def plot_template(
    projection_type,
    prefix_type,
    bin_times=[0.2, 0.4, 0.6, 0.8],
    alpha_required=0.6,
    alpha_optional=0.2,
    alpha_prefix=0.6,
    pad=0.02,
    plot=False,
):
    assert projection_type in [
        "shift",
        "bc_prediction",
    ], "projection type must be in ['shift', 'bc_prediction']"
    assert prefix_type in [
        "silence",
        "active",
        "overlap",
    ], "prefix type must be in ['silence', 'active', 'overlap']"

    colors = ["b", "orange"]
    current = 1.5
    bins = [current] + (np.array(bin_times).cumsum(0) + current).tolist()

    fig, ax = plt.subplots(1, 1)
    handles = []
    ax.hlines(y=0, xmin=0, xmax=bins[-1], linewidth=2, color="k")
    # Draw current line
    ax.vlines(current, ymin=-1, ymax=1, linewidth=5, color="r", label="current time")
    # current, = ax.plot([current, current], [-1, 1], linewidth=5, color="r", label='current time')
    # handles.append(current)

    #######################################################################
    # Draw prefix boxes
    #######################################################################
    if prefix_type == "silence":
        ax.add_patch(
            Rectangle(
                xy=(0, -1),
                width=1,
                height=1,
                facecolor=colors[1],
                alpha=alpha_prefix,
                edgecolor=colors[1],
            )
        )
    else:
        ax.add_patch(
            Rectangle(
                xy=(0, -1),
                width=1.5,
                height=1,
                facecolor=colors[1],
                alpha=alpha_prefix,
                edgecolor=colors[1],
            )
        )
        if prefix_type == "overlap":
            ax.add_patch(
                Rectangle(
                    xy=(1, 0),
                    width=0.5,
                    height=1,
                    facecolor=colors[0],
                    edgecolor=colors[0],
                )
            )

    #######################################################################
    # Projection Window Template
    #######################################################################
    if projection_type == "shift":
        # Optional
        optional_a = Rectangle(
            xy=(current, 0),
            width=bins[2] - bins[0],
            height=1,
            label="A optional",
            facecolor=colors[0],
            alpha=alpha_optional,
            edgecolor=colors[0],
        )
        handles.append(optional_a)
        ax.add_patch(optional_a)

        # Required
        required_a = Rectangle(
            xy=(bins[2], 0),
            width=bins[-1] - bins[2],
            height=1,
            label="A required",
            facecolor=colors[0],
            alpha=alpha_required,
            hatch=".",
            edgecolor=colors[0],
        )
        handles.append(required_a)
        ax.add_patch(required_a)
        if prefix_type != "silence":
            # Optional B
            optional_b = Rectangle(
                xy=(current, -1),
                width=bins[2] - bins[0],
                height=1,
                label="B optional",
                facecolor=colors[1],
                alpha=alpha_optional,
                edgecolor=colors[1],
            )
            handles.append(optional_b)
            ax.add_patch(optional_b)
    elif projection_type == "bc_prediction":
        # A at least one
        a_one = Rectangle(
            xy=(current, 0),
            width=bins[3] - bins[0],
            height=1,
            label="A at least 1",
            facecolor=colors[0],
            alpha=0.2,
            edgecolor=colors[0],
            hatch="//",
        )
        handles.append(a_one)
        ax.add_patch(a_one)
        # B optional
        optional_b = Rectangle(
            xy=(current, -1),
            width=bins[3] - bins[0],
            height=1,
            label="B optional",
            facecolor=colors[1],
            alpha=alpha_optional,
            edgecolor=colors[1],
        )
        handles.append(optional_b)
        ax.add_patch(optional_b)

        # B required
        required_b = Rectangle(
            xy=(bins[3], -1),
            width=bins[-1] - bins[3],
            height=1,
            label="B required",
            facecolor=colors[1],
            alpha=alpha_required,
            hatch=".",
            edgecolor=colors[1],
        )
        handles.append(required_b)
        ax.add_patch(required_b)

    # Draw lines for projection window template
    ax.vlines(bins[1:-1], ymin=-1, ymax=1, linewidth=2, color="k")
    ax.hlines(y=[-1, 1], xmin=1.5, xmax=bins[-1], linewidth=2, color="k")
    ax.vlines(bins[-1], ymin=-1, ymax=1, linewidth=2, color="k")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim([0, bins[-1] + 0.05])
    ax.legend(loc="upper left", handles=handles)

    plt.subplots_adjust(
        left=pad, bottom=pad, right=1 - pad, top=1 - pad, wspace=None, hspace=None
    )
    if plot:
        plt.pause(0.1)
    return fig, ax
