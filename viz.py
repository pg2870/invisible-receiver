"""
Visualizations for "The Invisible Receiver"
============================================
Reads the CSVs produced by invisible_receiver.py and writes two PNGs to
output/:

  leaderboard_top15.png
      Horizontal bar chart of the top-15 players by Invisible Rate, with Odell
      Beckham Jr. highlighted.

  play_diagram.png
      Bird's-eye snapshot of OBJ's single most-open invisible play at the throw
      frame. Offense is blue, defense red, the football black. OBJ is starred
      with a circle whose radius is his separation; the targeted receiver is
      labeled with his (much smaller) separation. The line of scrimmage and the
      first-down line are drawn. Coordinates are already normalized so the
      offense moves to the right.

Run (after invisible_receiver.py):  python viz.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                      # headless; just write PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

OBJ_NAME = "Odell Beckham"
OFFENSE_COLOR = "#1f77b4"     # blue
DEFENSE_COLOR = "#d62728"     # red
BALL_COLOR = "#111111"
OBJ_COLOR = "#2ca02c"         # green highlight
TARGET_COLOR = "#ff7f0e"      # orange

FIELD_WIDTH = 53.3


# --------------------------------------------------------------------------- #
# Chart 1 - top-15 leaderboard
# --------------------------------------------------------------------------- #
def leaderboard(path_out: str):
    df = pd.read_csv(os.path.join(OUT_DIR, "player_invisible_rate.csv"))
    top = df.sort_values("invisible_rate", ascending=False).head(15).copy()
    # plot lowest-to-highest so the biggest bar sits at the top
    top = top.iloc[::-1].reset_index(drop=True)
    top["pct"] = top["invisible_rate"] * 100

    is_obj = top["displayName"].str.contains(OBJ_NAME, case=False, na=False)
    colors = np.where(is_obj, OBJ_COLOR, "#9ecae1")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["displayName"], top["pct"], color=colors,
                   edgecolor="white")

    # value labels + route counts
    for bar, routes, inv in zip(bars, top["routes"], top["invisible"]):
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%  ({int(inv)}/{int(routes)})",
                va="center", ha="left", fontsize=9)

    # bold the OBJ tick label
    for lbl, flag in zip(ax.get_yticklabels(), is_obj):
        if flag:
            lbl.set_fontweight("bold")
            lbl.set_color(OBJ_COLOR)

    ax.set_xlabel("Invisible Rate  (open, un-targeted routes / routes)  %")
    ax.set_title("The Invisible Receiver - Top 15 by Invisible Rate\n"
                 "(min 100 scored routes, 2021 Weeks 1-8)", fontsize=13)
    ax.set_xlim(0, top["pct"].max() * 1.25)
    ax.margins(y=0.01)

    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=OBJ_COLOR,
               markersize=12, label="Odell Beckham Jr."),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9ecae1",
               markersize=12, label="Other receivers"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    print(f"    wrote {path_out}")


# --------------------------------------------------------------------------- #
# Chart 2 - OBJ most-open play diagram
# --------------------------------------------------------------------------- #
def play_diagram(path_out: str):
    frame = pd.read_csv(os.path.join(OUT_DIR, "obj_top_play_frame.csv"))

    los = float(frame["los_x"].iloc[0])
    ytg = float(frame["yardsToGo"].iloc[0])
    first_down = los + ytg

    offense = frame[frame["side"] == "offense"]
    defense = frame[frame["side"] == "defense"]
    ball = frame[frame["side"] == "ball"]
    obj = frame[frame["is_obj"]].iloc[0]
    tgt_rows = frame[frame["is_target"]]
    target = tgt_rows.iloc[0] if len(tgt_rows) else None

    # recompute each receiver's separation to nearest coverage defender so the
    # circle on the diagram is exact for this frame.
    cov = defense[defense["pff_role"] == "Coverage"]

    def nearest_def(px, py):
        if len(cov) == 0:
            return np.nan, None
        d = np.hypot(cov["x"] - px, cov["y"] - py)
        i = d.idxmin()
        return float(d.loc[i]), cov.loc[i]

    obj_sep, _ = nearest_def(obj["x"], obj["y"])
    tgt_sep = np.nan
    if target is not None:
        tgt_sep, _ = nearest_def(target["x"], target["y"])

    # ------- figure -------
    xmin = frame["x"].min() - 4
    xmax = frame["x"].max() + 4
    fig, ax = plt.subplots(figsize=(12, 6.2))

    # field backdrop + yard lines every 5 yd
    ax.set_facecolor("#eaf3ea")
    for xl in range(int(xmin // 5 * 5), int(xmax) + 5, 5):
        ax.axvline(xl, color="white", lw=1, zorder=0)

    # LOS and first-down line
    ax.axvline(los, color="#333333", lw=2, ls="-", zorder=1)
    ax.axvline(first_down, color="#f1c40f", lw=2, ls="--", zorder=1)
    ax.text(los, FIELD_WIDTH + 0.5, "LOS", color="#333333",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(first_down, FIELD_WIDTH + 0.5, "1st down", color="#b8860b",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # players
    ax.scatter(offense["x"], offense["y"], s=120, c=OFFENSE_COLOR,
               edgecolor="white", zorder=3, label="Offense")
    ax.scatter(defense["x"], defense["y"], s=120, c=DEFENSE_COLOR,
               edgecolor="white", zorder=3, label="Defense")
    if len(ball):
        ax.scatter(ball["x"], ball["y"], s=70, c=BALL_COLOR, marker="D",
                   zorder=4, label="Ball")

    # OBJ star + separation circle
    ax.scatter(obj["x"], obj["y"], s=520, marker="*", c=OBJ_COLOR,
               edgecolor="black", linewidth=0.8, zorder=6)
    if np.isfinite(obj_sep):
        circ = plt.Circle((obj["x"], obj["y"]), obj_sep, fill=False,
                          color=OBJ_COLOR, lw=2, ls="--", zorder=5)
        ax.add_patch(circ)
    ax.annotate(f"OBJ\nsep {obj_sep:.1f} yd",
                (obj["x"], obj["y"]),
                textcoords="offset points", xytext=(10, 12),
                fontsize=10, fontweight="bold", color=OBJ_COLOR, zorder=7)

    # targeted receiver label
    if target is not None:
        ax.scatter(target["x"], target["y"], s=200, marker="o",
                   facecolor="none", edgecolor=TARGET_COLOR, linewidth=2.5,
                   zorder=6)
        tname = str(target["displayName"]).split()[-1]
        ax.annotate(f"TARGET: {tname}\nsep {tgt_sep:.1f} yd",
                    (target["x"], target["y"]),
                    textcoords="offset points", xytext=(10, -28),
                    fontsize=10, fontweight="bold", color=TARGET_COLOR,
                    zorder=7)

    # title from the play
    gid = int(frame["gameId"].iloc[0])
    pid = int(frame["playId"].iloc[0])
    ax.set_title(
        f"OBJ's most-open invisible play - game {gid}, play {pid}\n"
        f"Open at {obj_sep:.1f} yd; QB threw to a receiver with {tgt_sep:.1f} yd "
        f"(throw frame, offense -> right)",
        fontsize=12,
    )
    ax.set_xlabel("Field x (yards, normalized so offense moves right)")
    ax.set_ylabel("Field y (yards)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-2, FIELD_WIDTH + 3)
    ax.set_aspect("equal", adjustable="box")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OFFENSE_COLOR,
               markersize=11, label="Offense"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DEFENSE_COLOR,
               markersize=11, label="Defense"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=OBJ_COLOR,
               markersize=16, label="OBJ (open, un-targeted)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=TARGET_COLOR, markersize=11,
               label="Targeted receiver"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=BALL_COLOR,
               markersize=9, label="Ball"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    print(f"    wrote {path_out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Building visualizations ...")
    leaderboard(os.path.join(OUT_DIR, "leaderboard_top15.png"))
    play_diagram(os.path.join(OUT_DIR, "play_diagram.png"))
    print("Done.")


if __name__ == "__main__":
    main()
