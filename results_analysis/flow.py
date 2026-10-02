"""Left-to-right flow charts of boxes connected by labelled arrows."""
from matplotlib import pyplot as plt


def draw_flow(
    nodes: dict,
    edges: list,
    title: str,
    note: str = "",
    figsize: tuple = (11.5, 6),
) -> None:
    """Draw a left-to-right flow chart of boxes connected by labelled arrows.

    nodes: key -> (x, y, text, facecolor, edgecolor), x/y in axes coordinates.
    edges: list of (source key, target key, arrow label) and, optionally, how
    far along the arrow the label sits (0 = at the source, 1 = at the target).
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = {
        key: ax.text(
            x, y, text,
            ha="left", va="center", fontsize=9.5, color=edge_colour,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=face,
                edgecolor=edge_colour,
                linewidth=1.2,
            ),
        )
        for key, (x, y, text, face, edge_colour) in nodes.items()
    }

    # The arrows start and end at the box borders, so the boxes have to be
    # laid out (drawn) before their extents are known.
    fig.canvas.draw()
    to_data = ax.transData.inverted().transform

    def border(key, side):
        extent = boxes[key].get_bbox_patch().get_window_extent()
        x0, y0 = to_data((extent.x0, extent.y0))
        x1, y1 = to_data((extent.x1, extent.y1))
        return (x1 if side == "right" else x0, (y0 + y1) / 2)

    for source, target, label, *rest in edges:
        at = rest[0] if rest else 0.5
        x0, y0 = border(source, "right")
        x1, y1 = border(target, "left")
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color="#666666", linewidth=1.3,
                shrinkA=3, shrinkB=3,
            ),
        )
        if label:
            ax.text(
                x0 + at * (x1 - x0), y0 + at * (y1 - y0), label,
                ha="center", va="center", fontsize=8.5, color="#444444",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
            )

    ax.set_title(title, fontsize=11.5, color="#222222", pad=14)
    if note:
        ax.text(
            0.0, -0.06, note, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color="#333333",
        )
    fig.tight_layout()
    plt.show()
    return None
