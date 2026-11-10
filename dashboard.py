"""
The Invisible Receiver - interactive dashboard
==============================================
Reads the pipeline outputs and writes a single self-contained HTML page to
docs/index.html (Plotly from CDN, no local assets needed).

Inputs (produced by invisible_receiver.py):
    output/player_invisible_rate.csv   - ranked player table
    output/obj_invisible_plays.csv     - OBJ's invisible plays, full context
    output/obj_top_play_frame.csv      - throw-frame positions, OBJ's top play

Output:
    docs/index.html

Run (after invisible_receiver.py):  python dashboard.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
DOCS_DIR = os.path.join(ROOT, "docs")

OBJ_NAME = "Odell Beckham"
OFFENSE_COLOR = "#1f77b4"     # blue
DEFENSE_COLOR = "#d62728"     # red
OBJ_COLOR = "#2ca02c"         # green
TARGET_COLOR = "#ff7f0e"      # orange
BAR_COLOR = "#9ecae1"
FIELD_WIDTH = 53.3


# --------------------------------------------------------------------------- #
# Chart 1 - top-15 invisible-rate bar chart
# --------------------------------------------------------------------------- #
def leaderboard_fig() -> go.Figure:
    df = pd.read_csv(os.path.join(OUT_DIR, "player_invisible_rate.csv"))
    top = (df.sort_values("invisible_rate", ascending=False)
           .head(15)
           .iloc[::-1]               # smallest first -> largest on top
           .reset_index(drop=True))
    top["pct"] = top["invisible_rate"] * 100
    is_obj = top["displayName"].str.contains(OBJ_NAME, case=False, na=False)
    colors = [OBJ_COLOR if o else BAR_COLOR for o in is_obj]

    custom = np.stack([top["routes"], top["invisible"], top["avg_depth"]], axis=-1)
    fig = go.Figure(
        go.Bar(
            x=top["pct"],
            y=top["displayName"],
            orientation="h",
            marker_color=colors,
            customdata=custom,
            text=[f"{v:.1f}%" for v in top["pct"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Invisible Rate: %{x:.1f}%<br>"
                "Routes: %{customdata[0]:.0f}<br>"
                "Invisible flags: %{customdata[1]:.0f}<br>"
                "Avg depth: %{customdata[2]:.1f} yd"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Top 15 by Invisible Rate (min 100 scored routes, 2021 Weeks 1-8)",
        xaxis_title="Invisible Rate  (open, un-targeted routes / routes)  %",
        yaxis_title="",
        margin=dict(l=160, r=60, t=60, b=50),
        height=560,
        template="plotly_white",
        bargap=0.25,
    )
    fig.update_xaxes(range=[0, top["pct"].max() * 1.2])
    return fig


# --------------------------------------------------------------------------- #
# Chart 2 - OBJ top-play field snapshot at the throw frame
# --------------------------------------------------------------------------- #
def play_fig() -> tuple[go.Figure, dict]:
    frame = pd.read_csv(os.path.join(OUT_DIR, "obj_top_play_frame.csv"))

    los = float(frame["los_x"].iloc[0])
    ytg = float(frame["yardsToGo"].iloc[0])
    first_down = los + ytg
    gid = int(frame["gameId"].iloc[0])
    pid = int(frame["playId"].iloc[0])

    offense = frame[(frame["side"] == "offense") & (~frame["is_obj"])
                    & (~frame["is_target"])]
    defense = frame[frame["side"] == "defense"]
    obj = frame[frame["is_obj"]].iloc[0]
    tgt_rows = frame[frame["is_target"]]
    target = tgt_rows.iloc[0] if len(tgt_rows) else None

    # exact separations at this frame (nearest coverage defender)
    cov = defense[defense["pff_role"] == "Coverage"]

    def nearest(px, py):
        if cov.empty:
            return np.nan
        return float(np.min(np.hypot(cov["x"] - px, cov["y"] - py)))

    obj_sep = nearest(obj["x"], obj["y"])
    tgt_sep = nearest(target["x"], target["y"]) if target is not None else np.nan

    fig = go.Figure()

    # yard lines every 5 yards
    xmin, xmax = frame["x"].min() - 4, frame["x"].max() + 4
    for xl in range(int(xmin // 5 * 5), int(xmax) + 5, 5):
        fig.add_vline(x=xl, line_width=1, line_color="#e5efe5")

    # LOS + first-down lines
    fig.add_vline(x=los, line_width=2, line_color="#333333",
                  annotation_text="LOS", annotation_position="top")
    fig.add_vline(x=first_down, line_width=2, line_dash="dash",
                  line_color="#d4a017",
                  annotation_text="1st down", annotation_position="top")

    # offense (non-OBJ, non-target)
    fig.add_trace(go.Scatter(
        x=offense["x"], y=offense["y"], mode="markers", name="Offense",
        marker=dict(size=13, color=OFFENSE_COLOR,
                    line=dict(width=1, color="white")),
        text=offense["displayName"],
        hovertemplate="%{text}<br>(%{x:.1f}, %{y:.1f})<extra>Offense</extra>",
    ))
    # defense
    fig.add_trace(go.Scatter(
        x=defense["x"], y=defense["y"], mode="markers", name="Defense",
        marker=dict(size=13, color=DEFENSE_COLOR,
                    line=dict(width=1, color="white")),
        text=defense["displayName"],
        hovertemplate="%{text}<br>(%{x:.1f}, %{y:.1f})<extra>Defense</extra>",
    ))
    # OBJ star + separation ring
    fig.add_trace(go.Scatter(
        x=[obj["x"]], y=[obj["y"]], mode="markers+text",
        name="OBJ (open, un-targeted)",
        marker=dict(size=24, color=OBJ_COLOR, symbol="star",
                    line=dict(width=1, color="black")),
        text=[f"  OBJ ({obj_sep:.1f} yd)"], textposition="top center",
        textfont=dict(color=OBJ_COLOR, size=13),
        hovertemplate=f"OBJ<br>separation {obj_sep:.1f} yd<extra></extra>",
    ))
    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=obj["x"] - obj_sep, x1=obj["x"] + obj_sep,
                  y0=obj["y"] - obj_sep, y1=obj["y"] + obj_sep,
                  line=dict(color=OBJ_COLOR, width=2, dash="dash"))
    # target
    if target is not None:
        tname = str(target["displayName"]).split()[-1]
        fig.add_trace(go.Scatter(
            x=[target["x"]], y=[target["y"]], mode="markers+text",
            name="Targeted receiver",
            marker=dict(size=16, color="rgba(0,0,0,0)", symbol="circle-open",
                        line=dict(width=3, color=TARGET_COLOR)),
            text=[f"  TARGET: {tname} ({tgt_sep:.1f} yd)"],
            textposition="bottom center",
            textfont=dict(color=TARGET_COLOR, size=13),
            hovertemplate=(f"Target: {target['displayName']}<br>"
                           f"separation {tgt_sep:.1f} yd<extra></extra>"),
        ))

    fig.update_layout(
        title=(f"OBJ's most-open invisible play - game {gid}, play {pid}"
               f"  (throw frame, offense -> right)"),
        xaxis_title="Field x (yards, normalized)",
        yaxis_title="Field y (yards)",
        template="plotly_white",
        height=560,
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(range=[xmin, xmax])
    fig.update_yaxes(range=[-2, FIELD_WIDTH + 3], scaleanchor="x", scaleratio=1)

    meta = dict(gid=gid, pid=pid, obj_sep=obj_sep, tgt_sep=tgt_sep,
                target=None if target is None else target["displayName"])
    return fig, meta


# --------------------------------------------------------------------------- #
# Table - OBJ's invisible plays
# --------------------------------------------------------------------------- #
def obj_table_html() -> str:
    df = pd.read_csv(os.path.join(OUT_DIR, "obj_invisible_plays.csv"))
    show = df.copy()
    show["separation"] = show["separation"].round(1)
    show["depth"] = show["depth"].round(1)
    show["target_separation"] = show["target_separation"].round(1)
    show = show[["week", "quarter", "down", "yardsToGo", "separation", "depth",
                 "target_name", "target_separation", "playDescription"]]
    show.columns = ["Wk", "Qtr", "Down", "ToGo", "OBJ sep", "Depth",
                    "Target", "Target sep", "Play description"]
    return show.to_html(index=False, border=0, justify="center",
                        classes="obj-table", escape=True)


# --------------------------------------------------------------------------- #
# Assemble the page
# --------------------------------------------------------------------------- #
def build():
    os.makedirs(DOCS_DIR, exist_ok=True)

    bar = leaderboard_fig()
    field, meta = play_fig()
    table = obj_table_html()
    n_plays = pd.read_csv(os.path.join(OUT_DIR, "obj_invisible_plays.csv")).shape[0]

    # first figure embeds the CDN script tag; the rest reuse it
    bar_html = to_html(bar, include_plotlyjs="cdn", full_html=False)
    field_html = to_html(field, include_plotlyjs=False, full_html=False)

    tgt = meta["target"] or "the actual target"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Invisible Receiver</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          margin: 0; padding: 0 0 48px; color: #1b1b1b; background: #fafafa; }}
  header {{ background: #11321f; color: #fff; padding: 28px 32px; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; }}
  header p {{ margin: 0; font-size: 15px; opacity: .9; max-width: 900px; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 20px; }}
  section {{ background: #fff; border: 1px solid #e7e7e7; border-radius: 8px;
             margin: 22px 0; padding: 14px 18px; }}
  section h2 {{ font-size: 19px; margin: 6px 0 2px; }}
  section p.note {{ color: #555; font-size: 13px; margin: 0 0 8px; }}
  table.obj-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.obj-table th, table.obj-table td {{ padding: 6px 8px; border-bottom: 1px solid #eee;
                                             text-align: center; }}
  table.obj-table th {{ background: #11321f; color: #fff; position: sticky; top: 0; }}
  table.obj-table td:last-child {{ text-align: left; }}
  .scroll {{ max-height: 460px; overflow: auto; }}
  footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 26px; }}
</style>
</head>
<body>
<header>
  <h1>The Invisible Receiver</h1>
  <p>Finding wide receivers and tight ends who got open on their routes but were
     not thrown the ball &mdash; measured as the share of a player's routes where
     he was sustained-open, more open than the actual target, and still ignored.</p>
</header>

<div class="wrap">
  <section>
    <h2>Who gets ignored while open</h2>
    <p class="note">Interactive &mdash; hover a bar for routes, invisible flags,
       and average route depth. Odell Beckham Jr. is highlighted in green.</p>
    {bar_html}
  </section>

  <section>
    <h2>Anatomy of a miss</h2>
    <p class="note">OBJ's single most-open invisible play at the moment of the
       throw. He was open by {meta['obj_sep']:.1f} yards; the quarterback instead
       threw to {tgt} with only {meta['tgt_sep']:.1f} yards of separation.</p>
    {field_html}
  </section>

  <section>
    <h2>OBJ's {n_plays} invisible plays</h2>
    <p class="note">Every play flagged invisible for OBJ, sorted by his
       separation at the throw.</p>
    <div class="scroll">{table}</div>
  </section>

  <footer>Built from NFL Big Data Bowl 2022 tracking data (Weeks 1-8, 2021).
    Descriptive separation metric, not a causal expected-value model.</footer>
</div>
</body>
</html>
"""
    out_path = os.path.join(DOCS_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {out_path}  ({os.path.getsize(out_path):,} bytes)")
    return out_path


if __name__ == "__main__":
    build()
