## Visualize the SLALOM scores.
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import numpy as np
import typing as tp


def slalom_scatter_plot(
    expl_list: tp.List[tp.Tuple[str, np.ndarray]],
    sizex=12,
    sizey=12,
    ylim=None,
    highlight_toks=[],
    fontsize=6,
):
    """Create a importance / value score scatter plot.
    :param: expl_list: List of slalom explanations for each token (list of tupes) The first two scores need to be the value (at index 0)
        and importance scores (at index 1)
    :param: sizex: width of the plot
    :param: sizey: height of the plot
    :param: ylim: the ylimits, pass a tuple or 2-element list, e.g. [3,6]
    :param: highlight_toks: specifically highlight only some tokens, e.g. important words.
    """
    tok_list = []
    fig, ax = plt.subplots()
    plotted_set = set()
    for tok_str, slalom_scores in expl_list:
        if tok_str in plotted_set:
            continue
        tok_str = tok_str.replace("Ġ", "").replace(
            "##", ""
        )  ## For GPT2-tokens and subword tokens
        if tok_str in highlight_toks:
            plt.scatter([slalom_scores[0]], [slalom_scores[1]], 3, c="tab:red")
            plt.annotate(
                tok_str, xy=(slalom_scores[0], slalom_scores[1]), fontsize=fontsize
            )
        else:
            plt.scatter([slalom_scores[0]], [slalom_scores[1]], 3, c="tab:blue")
            if len(highlight_toks) == 0:
                plt.annotate(
                    tok_str, xy=(slalom_scores[0], slalom_scores[1]), fontsize=fontsize
                )
        plotted_set.add(tok_str)

    plt.xlabel("value score")
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel("importance score")
    plt.gcf().set_size_inches(sizey, sizex)
    return fig, ax


def slalom_highlight_plot(
    expl_list: tp.List[tp.Tuple[str, np.ndarray]], max_len=None, start_pos_y=500, vmax=5
):
    """Create the standard feature attribution plot, highlighting the importance of each word.
    :param: expl_list: List of slalom explanations for each token (list of tupes).
        The first score (or a single number) for each token will be visualized.
    :param: max_len: how many tokens should be visualized.
    :param: start_pos_y: for visalizing multiple rows, how high the plot will be(the uppermost position will contain the first row)
    :param: vmax: Maximimun value for normalizing the attribution.

    """
    fig, ax = plt.subplots(figsize=(7, 4.9))
    rend = fig.canvas.get_renderer()
    pos_x = 15
    max_x = 500
    pos_y = start_pos_y
    whitespace = 10
    norm = Normalize(vmin=-vmax, vmax=vmax)
    # cmap = plt.cm.YlOrRd
    # cmap = plt.cm.bwr
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red", "white", "green"])
    plt.xlim([0, max_x + 50])
    expl_list = expl_list[:max_len]
    for token, att in expl_list:
        if isinstance(att, np.ndarray):
            att = att[0]
        bb = dict(boxstyle="square, pad=0.2", fc=cmap(norm(att)), alpha=0.6)
        text = plt.text(
            pos_x,
            pos_y,
            token.replace("##", "").replace("Ġ", ""),
            color="black",
            bbox=bb,
            fontsize=12,
        )
        pos_x += text.get_window_extent(renderer=rend).width + whitespace

        if pos_x > max_x:
            pos_x = 15
            pos_y -= 25
    plt.ylim([pos_y - 40, start_pos_y + 15])
    # cax = fig.add_axes([0.2, 0.0, 0.6, 0.05])  # Adjust these values as needed
    # cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    # cb.set_label('Attention Score', fontsize=10)
    # cb.ax.tick_params(axis='both', labelsize=10)
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    plt.show()
    return fig, ax
