"""
The Invisible Receiver
=======================
Identify WRs/TEs who repeatedly get open on routes but are not targeted.

Pipeline
--------
1. Load plays / players / pffScoutingData; keep passResult in {C, I, IN}.
2. Stream the per-game tracking files, keeping only two frames per play:
      - the throw frame   : first of pass_forward / autoevent_passforward
      - the arrival frame : first of pass_arrived / pass_outcome_caught /
                            pass_outcome_incomplete
3. Normalize x so the offense always moves to the right (x -> 120 - x when
   playDirection == 'left'); mirror y as well so geometry stays consistent.
4. Target = the WR/TE/RB route runner (pff_role == 'Pass Route') for the play.
   Primary method: the arrival-frame ball is only tagged on ~5% of plays in
   this dataset, so the target is resolved by parsing the receiver named in
   playDescription and matching to the route runners on that play. Where an
   arrival frame WITH a ball row does exist, we cross-check against the
   nearest route runner to the football. Validate 20 plays vs. playDescription.
5. For every WR/TE route runner, compute separation (distance to the nearest
   Coverage defender) in EACH frame of the window throw-5..throw. Derive:
      - separation     : at the throw frame
      - min_sep_window : minimum separation across the 6 frames
      - sustained_open : min_sep_window >= 3 yd
      - depth          : yards past the LOS at the throw frame
      - qb_dist        : QB-to-receiver distance at the throw frame
6. invisible = not target AND sustained_open AND
   separation >= target_separation + 1.5 AND depth in [0,25] AND qb_dist <= 30.
   Also: clean_pocket (no rusher with a hit/hurry/sack on the play) and
   costly_miss (invisible AND the play stalled: incomplete, or 3rd/4th down
   gaining fewer yards than needed).
7. Per player: routes, open_rate, ignored_when_open, invisible_rate with a
   Wilson 95% CI, costly_miss count, avg_depth, past_sticks_invisible_count.
   Reported for all plays AND clean-pocket plays. (min 100 routes)
8. Per passer (pff_role == 'Pass'): invisible routes / plays -> missed-open
   rate (min 100 plays).
9. Export Odell Beckham Jr.'s invisible plays with full context, and the
   throw-frame positions of the single most-open OBJ play (for viz.py).
10. Sensitivity: rerun the flag at separation thresholds 2.5/3.0/3.5 and record
    OBJ's rank and rate at each -> sensitivity_obj.csv.

Run:  python invisible_receiver.py
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRACKING_GLOB = os.path.join(DATA_DIR, "tracking", "tracking_*.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

FIELD_LENGTH = 120.0          # x axis, yards (incl. both end zones)
FIELD_WIDTH = 53.3            # y axis, yards

THROW_EVENTS = {"pass_forward", "autoevent_passforward"}
ARRIVAL_EVENTS = {"pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete"}

TARGET_POSITIONS = {"WR", "TE", "RB"}   # route runners eligible to be "target"
RECEIVER_POSITIONS = {"WR", "TE"}       # population we score for invisibility

SEP_MIN = 3.0            # yards of separation to count as "open" / sustained
SEP_GAP = 1.5            # receiver must beat the target's separation by this
DEPTH_MIN = 0.0          # depth window past LOS
DEPTH_MAX = 25.0
QB_DIST_MAX = 30.0       # max QB-to-receiver distance at the throw (yards)
WINDOW = 5               # frames before the throw to require sustained openness
MIN_ROUTES = 100         # min routes for a player to appear in the rate table
MIN_QB_PLAYS = 100       # min plays for a passer to appear in the QB table

SENS_THRESHOLDS = [2.5, 3.0, 3.5]   # separation sweep for sensitivity analysis

OBJ_NAME = "Odell Beckham"   # matched via players.displayName


# --------------------------------------------------------------------------- #
# Step 1 - load static tables
# --------------------------------------------------------------------------- #
def load_static():
    print("[1] Loading plays / players / pffScoutingData ...")

    # NA tokens in this dataset are the literal string "NA".
    na = ["NA", "", "NaN"]

    plays = pd.read_csv(
        os.path.join(DATA_DIR, "plays.csv"),
        na_values=na,
        usecols=[
            "gameId", "playId", "playDescription", "passResult",
            "quarter", "down", "yardsToGo", "playResult",
            "possessionTeam", "defensiveTeam",
            "absoluteYardlineNumber", "pff_passCoverageType",
        ],
    )
    plays = plays[plays["passResult"].isin(["C", "I", "IN"])].copy()

    # week comes from games.csv; attach it to each play for later context
    games = pd.read_csv(
        os.path.join(DATA_DIR, "games.csv"),
        na_values=na,
        usecols=["gameId", "week"],
    )
    plays = plays.merge(games, on="gameId", how="left")

    players = pd.read_csv(
        os.path.join(DATA_DIR, "players.csv"),
        na_values=na,
        usecols=["nflId", "officialPosition", "displayName"],
    )

    pff = pd.read_csv(
        os.path.join(DATA_DIR, "pffScoutingData.csv"),
        na_values=na,
        usecols=["gameId", "playId", "nflId", "pff_role", "pff_blockType",
                 "pff_hit", "pff_hurry", "pff_sack"],
    )

    # passer per play: the single player tagged pff_role == 'Pass'
    passers = pff[pff["pff_role"] == "Pass"][["gameId", "playId", "nflId"]].copy()
    passers = passers.merge(
        players[["nflId", "displayName"]], on="nflId", how="left"
    ).rename(columns={"nflId": "passer_nflId", "displayName": "passer_name"})
    passers = passers.drop_duplicates(["gameId", "playId"])

    # clean pocket: no pass rusher on the play recorded a hit / hurry / sack.
    rush = pff[pff["pff_role"] == "Pass Rush"].copy()
    rush["pressure"] = (
        rush[["pff_hit", "pff_hurry", "pff_sack"]].eq(1).any(axis=1)
    )
    pressured = (
        rush.groupby(["gameId", "playId"])["pressure"].any()
        .rename("pressured").reset_index()
    )
    plays = plays.merge(pressured, on=["gameId", "playId"], how="left")
    plays["pressured"] = plays["pressured"].fillna(False).astype(bool)
    plays["clean_pocket"] = (~plays["pressured"]).astype(bool)

    print(f"    plays kept (C/I/IN): {len(plays):,}")
    print(f"    clean-pocket plays: {int(plays['clean_pocket'].sum()):,}")
    print(f"    players           : {len(players):,}")
    print(f"    pff rows          : {len(pff):,}")
    print(f"    passers (1/play)  : {len(passers):,}")
    return plays, players, pff, passers


# --------------------------------------------------------------------------- #
# Step 2 - stream tracking files, keep the throw window + arrival frame
# --------------------------------------------------------------------------- #
def load_tracking_frames(valid_keys: set[tuple[int, int]]):
    """Return rows for two things per play:
      * the throw WINDOW: every player's rows for frames [throw-5 .. throw],
        where `throw` is the first throw-event frame. frame_kind == 'window',
        with `frame_offset` in -5..0 (0 == throw frame).
      * the arrival frame: the first arrival-event frame. frame_kind=='arrival'.
    """
    print("[2] Scanning tracking files for throw-window / arrival frames ...")
    files = sorted(glob.glob(TRACKING_GLOB))
    if not files:
        raise FileNotFoundError(f"No tracking files matched {TRACKING_GLOB}")

    usecols = [
        "gameId", "playId", "nflId", "frameId", "event",
        "x", "y", "playDirection", "team",
    ]
    keys = pd.DataFrame(list(valid_keys), columns=["gameId", "playId"])
    parts = []

    for i, path in enumerate(files, 1):
        df = pd.read_csv(path, usecols=usecols, na_values=["NA", "", "NaN"])
        df = df.merge(keys, on=["gameId", "playId"], how="inner")
        if df.empty:
            continue

        # ---- locate the throw frame per play (first throw event) ----
        thr = df[df["event"].isin(THROW_EVENTS)]
        thr_frame = (thr.groupby(["gameId", "playId"])["frameId"].min()
                     .rename("throw_frame").reset_index())

        # ---- locate the arrival frame per play (first arrival event) ----
        arr = df[df["event"].isin(ARRIVAL_EVENTS)]
        arr_frame = (arr.groupby(["gameId", "playId"])["frameId"].min()
                     .rename("arrival_frame").reset_index())

        df = df.merge(thr_frame, on=["gameId", "playId"], how="left")
        df = df.merge(arr_frame, on=["gameId", "playId"], how="left")

        # throw window: frames within [throw-5, throw]
        off = df["frameId"] - df["throw_frame"]
        in_window = df["throw_frame"].notna() & off.between(-5, 0)
        win = df[in_window].copy()
        win["frame_kind"] = "window"
        win["frame_offset"] = (win["frameId"] - win["throw_frame"]).astype(int)

        # arrival frame rows
        is_arr = df["arrival_frame"].notna() & (df["frameId"] == df["arrival_frame"])
        arrdf = df[is_arr].copy()
        arrdf["frame_kind"] = "arrival"
        arrdf["frame_offset"] = np.nan

        keep = pd.concat([win, arrdf], ignore_index=True)
        parts.append(keep.drop(columns=["throw_frame", "arrival_frame"]))

        if i % 20 == 0 or i == len(files):
            print(f"    processed {i}/{len(files)} files")

    frames = pd.concat(parts, ignore_index=True)
    n_plays = frames.groupby(["gameId", "playId"]).ngroups
    print(f"    tracking rows retained: {len(frames):,} across {n_plays:,} plays")
    return frames


# --------------------------------------------------------------------------- #
# Step 3 - normalize orientation so offense always moves right
# --------------------------------------------------------------------------- #
def normalize_orientation(frames: pd.DataFrame) -> pd.DataFrame:
    print("[3] Normalizing coordinates (offense -> right) ...")
    left = frames["playDirection"].eq("left")
    frames["x"] = np.where(left, FIELD_LENGTH - frames["x"], frames["x"])
    frames["y"] = np.where(left, FIELD_WIDTH - frames["y"], frames["y"])
    return frames


# --------------------------------------------------------------------------- #
# Helper - attach roles / positions / play context to tracking rows
# --------------------------------------------------------------------------- #
def enrich(frames, pff, players, plays):
    f = frames.merge(pff, on=["gameId", "playId", "nflId"], how="left")
    f = f.merge(players, on="nflId", how="left")
    f = f.merge(
        plays[["gameId", "playId", "possessionTeam", "defensiveTeam",
               "absoluteYardlineNumber", "yardsToGo", "pff_passCoverageType",
               "clean_pocket", "down", "playResult", "passResult"]],
        on=["gameId", "playId"], how="left",
    )
    # absoluteYardlineNumber lives in the RAW coordinate frame (it tracks raw x
    # of the line of scrimmage). We flipped x -> 120 - x for left-moving plays,
    # so flip the LOS the same way to keep both on the offense-moves-right axis.
    left = f["playDirection"].eq("left")
    f["los_x"] = np.where(left, FIELD_LENGTH - f["absoluteYardlineNumber"],
                          f["absoluteYardlineNumber"])
    f["is_ball"] = f["team"].eq("football")
    f["is_route"] = f["pff_role"].eq("Pass Route")
    f["is_coverage"] = f["pff_role"].eq("Coverage")
    return f


# --------------------------------------------------------------------------- #
# Step 4 - target identification
# --------------------------------------------------------------------------- #
def _desc_target_token(desc: str) -> str | None:
    """Pull the targeted receiver token ('C.Godwin') from a play description.

    The targeted receiver is the name that follows 'pass ... to <Name>'. We take
    the last such 'to X.Lastname' token because the receiver is named at the end
    of the pass clause (yardage / tacklers come after)."""
    if not isinstance(desc, str):
        return None
    matches = re.findall(r"\bto\s+([A-Z][A-Za-z'\-]*\.[A-Z][A-Za-z'\-]+)", desc)
    if not matches:
        return None
    return matches[-1]                  # e.g. 'C.Godwin'


def _norm_last(name: str) -> str:
    return str(name).split(".")[-1].split()[-1].lower().replace("'", "")


def _desc_last(desc: str) -> str | None:
    tok = _desc_target_token(desc)
    return _norm_last(tok) if tok else None


def _desc_initial(desc: str) -> str | None:
    tok = _desc_target_token(desc)
    return tok.split(".", 1)[0][0].lower() if tok else None


def identify_targets(arrival, throw, plays, players):
    """Resolve the target per play by matching the receiver named in
    playDescription to the route runners on that play. Ball-proximity at the
    arrival frame is used only as a cross-check (it exists on ~5% of plays)."""
    print("[4] Identifying targeted receiver (playDescription -> route runner) ...")

    # route runners eligible to be the target, per play, with name parts
    routers = pd.concat([arrival, throw], ignore_index=True)
    routers = routers[
        routers["is_route"] & routers["officialPosition"].isin(TARGET_POSITIONS)
    ][["gameId", "playId", "nflId", "displayName"]].drop_duplicates()
    routers["last"] = routers["displayName"].map(_norm_last)
    routers["initial"] = routers["displayName"].str[0].str.lower()

    desc = plays[["gameId", "playId", "playDescription"]].copy()
    desc["t_last"] = desc["playDescription"].map(_desc_last)
    desc["t_init"] = desc["playDescription"].map(_desc_initial)

    cand = routers.merge(
        desc[["gameId", "playId", "t_last", "t_init"]],
        on=["gameId", "playId"], how="inner",
    )
    # match on last name; disambiguate same-surname teammates by first initial
    cand["last_match"] = cand["last"] == cand["t_last"]
    cand["init_match"] = cand["initial"] == cand["t_init"]
    hits = cand[cand["last_match"]].copy()

    # prefer rows where first initial also matches (resolves e.g. two Smiths)
    hits["score"] = hits["init_match"].astype(int)
    hits = hits.sort_values(["gameId", "playId", "score"], ascending=[True, True, False])
    target = hits.groupby(["gameId", "playId"], as_index=False).first()

    out = target[["gameId", "playId", "nflId", "displayName"]].rename(
        columns={"nflId": "target_nflId", "displayName": "target_name"}
    )
    print(f"    plays with an identified target: {len(out):,} "
          f"of {len(plays):,} C/I/IN plays")
    return out


# --------------------------------------------------------------------------- #
# Step 4b - validate 20 plays against playDescription
# --------------------------------------------------------------------------- #
def validate_targets(targets, arrival, plays, n=20):
    """Independent check: compare the resolved target to (a) the surname in the
    description and (b) the route runner nearest the ball on plays that actually
    have an arrival-frame ball row."""
    print(f"[4b] Validating {n} targets against playDescription ...")
    m = targets.merge(
        plays[["gameId", "playId", "playDescription", "passResult"]],
        on=["gameId", "playId"], how="left",
    )
    m["desc_last"] = m["playDescription"].map(_desc_last)
    m = m[m["desc_last"].notna()].copy()
    m["pred_last"] = m["target_name"].map(_norm_last)
    m["match"] = m["pred_last"] == m["desc_last"]

    sample = m.head(n)
    hits = int(sample["match"].sum())
    print(f"    sample matched: {hits}/{len(sample)}")
    for _, r in sample.iterrows():
        flag = "OK " if r["match"] else "MISS"
        print(f"      [{flag}] pred={r['target_name']:<22} "
              f"desc~{r['desc_last']:<14} ({r['passResult']})")
    overall = m["match"].mean() if len(m) else float("nan")
    print(f"    full-set target vs description match: {overall:.1%} "
          f"on {len(m):,} named-receiver plays")

    # geometric cross-check on plays that do have an arrival-frame ball
    ball = arrival[arrival["is_ball"]][["gameId", "playId", "x", "y"]].rename(
        columns={"x": "bx", "y": "by"}).drop_duplicates(["gameId", "playId"])
    cand = arrival[
        arrival["is_route"] & arrival["officialPosition"].isin(TARGET_POSITIONS)
    ].merge(ball, on=["gameId", "playId"], how="inner")
    if len(cand):
        cand["d"] = np.hypot(cand["x"] - cand["bx"], cand["y"] - cand["by"])
        nearest = (cand.sort_values(["gameId", "playId", "d"])
                   .groupby(["gameId", "playId"], as_index=False).first()
                   [["gameId", "playId", "nflId"]]
                   .rename(columns={"nflId": "nearest_nflId"}))
        chk = targets.merge(nearest, on=["gameId", "playId"], how="inner")
        agree = (chk["target_nflId"] == chk["nearest_nflId"]).mean()
        print(f"    description-target vs nearest-to-ball agreement: "
              f"{agree:.1%} on {len(chk):,} arrival-ball plays")
    return sample


# --------------------------------------------------------------------------- #
# Step 5 - per-frame separation over the throw window + throw-frame metrics
# --------------------------------------------------------------------------- #
def _nearest_cov_per_frame(window):
    """For every (play, receiver, frame) return distance to the nearest
    coverage defender in that same frame."""
    recs = window[
        window["is_route"] & window["officialPosition"].isin(RECEIVER_POSITIONS)
    ][["gameId", "playId", "nflId", "frameId", "x", "y"]]

    defs = window[window["is_coverage"]][
        ["gameId", "playId", "frameId", "x", "y"]
    ].rename(columns={"x": "def_x", "y": "def_y"})

    # match within the same frame of the same play
    merged = recs.merge(defs, on=["gameId", "playId", "frameId"], how="left")
    merged["d"] = np.hypot(merged["x"] - merged["def_x"],
                           merged["y"] - merged["def_y"])
    sep = (merged.groupby(["gameId", "playId", "nflId", "frameId"],
                          as_index=False)["d"].min()
           .rename(columns={"d": "sep"}))
    return sep


def separation_over_window(window: pd.DataFrame) -> pd.DataFrame:
    """One row per scored receiver-route with:
        separation        - at the throw frame (offset 0)
        min_sep_window     - min separation across frames throw-5..throw
        sustained_open     - min_sep_window >= SEP_MIN
        depth              - downfield yards past LOS at the throw frame
        qb_dist            - QB-to-receiver distance at the throw frame
    """
    print("[5] Computing per-frame separation across the throw window ...")

    per_frame = _nearest_cov_per_frame(window)

    # min separation across all window frames, plus the frame count (should be
    # up to 6; sustained_open only trusts plays with the full window present).
    agg = (per_frame.groupby(["gameId", "playId", "nflId"], as_index=False)
           .agg(min_sep_window=("sep", "min"),
                n_window_frames=("sep", "size")))

    # throw-frame slice (offset 0) carries all the static context we need
    throw = window[window["frame_offset"] == 0].copy()
    receivers = throw[
        throw["is_route"] & throw["officialPosition"].isin(RECEIVER_POSITIONS)
    ][["gameId", "playId", "nflId", "displayName", "officialPosition",
       "x", "y", "los_x", "yardsToGo", "pff_passCoverageType",
       "clean_pocket", "down", "playResult", "passResult",
       "pff_blockType"]].copy()

    # throw-frame separation = window separation at offset 0
    thr_sep = (per_frame.merge(
        throw[["gameId", "playId", "nflId", "frameId"]],
        on=["gameId", "playId", "nflId", "frameId"], how="inner")
        [["gameId", "playId", "nflId", "sep"]]
        .rename(columns={"sep": "separation"}))

    # QB position at the throw frame
    qb = throw[throw["pff_role"] == "Pass"][
        ["gameId", "playId", "x", "y"]
    ].rename(columns={"x": "qb_x", "y": "qb_y"}).drop_duplicates(
        ["gameId", "playId"])

    rec = (receivers
           .merge(thr_sep, on=["gameId", "playId", "nflId"], how="left")
           .merge(agg, on=["gameId", "playId", "nflId"], how="left")
           .merge(qb, on=["gameId", "playId"], how="left"))

    rec["depth"] = rec["x"] - rec["los_x"]
    rec["qb_dist"] = np.hypot(rec["x"] - rec["qb_x"], rec["y"] - rec["qb_y"])
    rec["sustained_open"] = (
        rec["min_sep_window"].notna() & (rec["min_sep_window"] >= SEP_MIN)
    )

    full = int((rec["n_window_frames"] == 6).sum())
    print(f"    receiver-routes scored : {len(rec):,} "
          f"({full:,} with a full 6-frame window)")
    print(f"    sustained-open routes  : {int(rec['sustained_open'].sum()):,}")
    return rec


# --------------------------------------------------------------------------- #
# Step 6 - flag invisible routes (new sustained-open definition)
# --------------------------------------------------------------------------- #
def attach_target_context(rec, targets):
    """Attach is_target, target_separation and past_sticks once up front."""
    rec = rec.merge(
        targets[["gameId", "playId", "target_nflId"]],
        on=["gameId", "playId"], how="left",
    )
    rec["is_target"] = rec["nflId"] == rec["target_nflId"]
    tgt_sep = (
        rec[rec["is_target"]][["gameId", "playId", "separation"]]
        .rename(columns={"separation": "target_separation"})
        .drop_duplicates(["gameId", "playId"])
    )
    rec = rec.merge(tgt_sep, on=["gameId", "playId"], how="left")
    rec["past_sticks"] = rec["depth"] >= rec["yardsToGo"]
    return rec


def compute_invisible(rec, sep_min=SEP_MIN, exclude_chip=True):
    """Return a boolean Series for the invisible flag at a given separation
    threshold. sustained_open is recomputed against sep_min so the sensitivity
    sweep is internally consistent.

    invisible =
        not the target
        AND sustained open  (min window separation >= sep_min)
        AND separation >= target_separation + SEP_GAP
        AND depth in [0, 25]
        AND QB-to-receiver distance <= QB_DIST_MAX at the throw
        AND (when exclude_chip) the route was NOT a chip block (pff_blockType
            != 'CH') - chippers sit uncovered at the LOS as safety valves, so
            their high separation is not a genuine missed opportunity. They stay
            in the route denominator; they just can't be flagged invisible.
    """
    sustained = rec["min_sep_window"].notna() & (rec["min_sep_window"] >= sep_min)
    flag = (
        (~rec["is_target"])
        & sustained
        & rec["separation"].notna()
        & rec["target_separation"].notna()
        & (rec["separation"] >= rec["target_separation"] + SEP_GAP)
        & (rec["depth"] >= DEPTH_MIN)
        & (rec["depth"] <= DEPTH_MAX)
        & rec["qb_dist"].notna()
        & (rec["qb_dist"] <= QB_DIST_MAX)
    )
    if exclude_chip:
        flag = flag & (rec["pff_blockType"] != "CH")
    return flag


def flag_invisible(rec: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    print("[6] Flagging invisible routes (sustained-open definition) ...")
    rec = attach_target_context(rec, targets)

    # with and without the chip exclusion, to report what the exclusion removed
    inv_with_chip = compute_invisible(rec, SEP_MIN, exclude_chip=False)
    rec["invisible"] = compute_invisible(rec, SEP_MIN, exclude_chip=True)
    chip_removed = int((inv_with_chip & ~rec["invisible"]).sum())

    # costly miss: an invisible route on a play that stalled.
    stalled_down = rec["down"].isin([3, 4]) & (rec["playResult"] < rec["yardsToGo"])
    incomplete = rec["passResult"].isin(["I", "IN"])
    rec["costly_miss"] = rec["invisible"] & (incomplete | stalled_down)

    n_all = int(rec["invisible"].sum())
    n_clean = int((rec["invisible"] & rec["clean_pocket"]).sum())
    n_costly = int(rec["costly_miss"].sum())
    print(f"    invisible before chip exclusion: {int(inv_with_chip.sum()):,}")
    print(f"    chip (CH) flags removed        : {chip_removed:,}")
    print(f"    invisible (all plays)    : {n_all:,}")
    print(f"    invisible (clean pocket) : {n_clean:,}")
    print(f"    costly misses            : {n_costly:,}")
    rec.attrs["chip_removed"] = chip_removed
    rec.attrs["invisible_before_chip"] = int(inv_with_chip.sum())
    return rec


# --------------------------------------------------------------------------- #
# Wilson score interval for a binomial proportion
# --------------------------------------------------------------------------- #
def wilson_ci(successes, n, z=1.96):
    """Vectorized Wilson 95% CI. Returns (low, high) arrays."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, successes / n, 0.0)
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    low = np.clip(center - half, 0, 1)
    high = np.clip(center + half, 0, 1)
    return low, high


# --------------------------------------------------------------------------- #
# Step 7 - per-player aggregation + coverage split
# --------------------------------------------------------------------------- #
def aggregate(rec: pd.DataFrame):
    print("[7] Aggregating per player ...")

    rec = rec.copy()
    rec["past_sticks_invisible"] = rec["invisible"] & rec["past_sticks"]
    # "ignored while open": the receiver was open (sustained) and not the target
    rec["ignored_open"] = rec["sustained_open"] & (~rec["is_target"])

    per_player = (
        rec.groupby(["nflId", "displayName", "officialPosition"], as_index=False)
        .agg(routes=("invisible", "size"),
             invisible=("invisible", "sum"),
             open_routes=("sustained_open", "sum"),
             ignored_open_count=("ignored_open", "sum"),
             costly_miss=("costly_miss", "sum"),
             avg_depth=("depth", "mean"),
             past_sticks_invisible_count=("past_sticks_invisible", "sum"))
    )
    per_player["invisible_rate"] = per_player["invisible"] / per_player["routes"]
    per_player["open_rate"] = per_player["open_routes"] / per_player["routes"]
    # share of a player's OPEN routes on which he was not the target
    per_player["ignored_when_open"] = np.where(
        per_player["open_routes"] > 0,
        per_player["ignored_open_count"] / per_player["open_routes"],
        np.nan,
    )
    lo, hi = wilson_ci(per_player["invisible"], per_player["routes"])
    per_player["invisible_rate_ci_low"] = lo
    per_player["invisible_rate_ci_high"] = hi
    per_player["avg_depth"] = per_player["avg_depth"].round(2)

    qualified = (
        per_player[per_player["routes"] >= MIN_ROUTES]
        .sort_values("invisible_rate", ascending=False)
        .reset_index(drop=True)
    )

    # coverage-type split (man / zone / other)
    cov = (
        rec.groupby(
            ["nflId", "displayName", "pff_passCoverageType"], as_index=False
        )
        .agg(routes=("invisible", "size"), invisible=("invisible", "sum"))
    )
    cov["invisible_rate"] = cov["invisible"] / cov["routes"]

    return per_player, qualified, cov


# --------------------------------------------------------------------------- #
# Step 10 - sensitivity sweep over the separation threshold
# --------------------------------------------------------------------------- #
def sensitivity_obj(rec, thresholds=SENS_THRESHOLDS):
    """Re-flag invisible at each separation threshold and record OBJ's rank and
    invisible rate among qualified players (>= MIN_ROUTES)."""
    print("[10] Sensitivity sweep over separation thresholds ...")
    rows = []
    for thr in thresholds:
        inv = compute_invisible(rec, sep_min=thr)
        tmp = rec.assign(inv=inv)
        agg = (tmp.groupby(["nflId", "displayName"], as_index=False)
               .agg(routes=("inv", "size"), invisible=("inv", "sum")))
        agg = agg[agg["routes"] >= MIN_ROUTES].copy()
        agg["invisible_rate"] = agg["invisible"] / agg["routes"]
        agg = agg.sort_values("invisible_rate", ascending=False).reset_index(drop=True)
        agg["rank"] = np.arange(1, len(agg) + 1)

        obj = agg[agg["displayName"].str.contains(OBJ_NAME, case=False, na=False)]
        if len(obj):
            r = obj.iloc[0]
            rows.append({
                "sep_threshold": thr,
                "obj_rank": int(r["rank"]),
                "n_qualified": len(agg),
                "obj_routes": int(r["routes"]),
                "obj_invisible": int(r["invisible"]),
                "obj_invisible_rate": round(float(r["invisible_rate"]), 4),
            })
        else:
            rows.append({
                "sep_threshold": thr, "obj_rank": np.nan,
                "n_qualified": len(agg), "obj_routes": np.nan,
                "obj_invisible": np.nan, "obj_invisible_rate": np.nan,
            })
    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        print(f"    sep>={r['sep_threshold']}:  OBJ rank "
              f"{r['obj_rank']}/{r['n_qualified']}  "
              f"rate {r['obj_invisible_rate']:.1%}")
    return out


# --------------------------------------------------------------------------- #
# Step 8 - per-passer missed-open rate
# --------------------------------------------------------------------------- #
def qb_missed_open(rec, targets, passers):
    """missed-open rate = total invisible routes thrown in front of a QB,
    divided by his pass plays. One play can contribute multiple invisible
    routes, so the ratio can exceed the share of plays with >=1 open man."""
    print("[8] Aggregating per passer (QB missed-open rate) ...")

    # invisible routes per play
    inv_per_play = (
        rec.groupby(["gameId", "playId"], as_index=False)["invisible"].sum()
        .rename(columns={"invisible": "invisible_routes"})
    )
    # restrict to plays that actually produced a scored target/receiver set
    scored_plays = targets[["gameId", "playId"]].drop_duplicates()
    pp = scored_plays.merge(inv_per_play, on=["gameId", "playId"], how="left")
    pp["invisible_routes"] = pp["invisible_routes"].fillna(0)
    pp = pp.merge(passers, on=["gameId", "playId"], how="left")

    qb = (
        pp.dropna(subset=["passer_nflId"])
        .groupby(["passer_nflId", "passer_name"], as_index=False)
        .agg(plays=("playId", "size"),
             invisible_routes=("invisible_routes", "sum"))
    )
    qb["qb_missed_open_rate"] = qb["invisible_routes"] / qb["plays"]
    qb = (
        qb[qb["plays"] >= MIN_QB_PLAYS]
        .sort_values("qb_missed_open_rate", ascending=False)
        .reset_index(drop=True)
    )
    print(f"    qualified passers (>= {MIN_QB_PLAYS} plays): {len(qb):,}")
    return qb


# --------------------------------------------------------------------------- #
# Step 9 - Odell Beckham Jr. exports
# --------------------------------------------------------------------------- #
def obj_exports(rec, throw, players, plays):
    """Write OBJ's invisible plays (full context) and the throw-frame positions
    of his single most-open invisible play, for the play diagram in viz.py."""
    print("[9] Building Odell Beckham Jr. exports ...")

    obj = players[players["displayName"].str.contains(OBJ_NAME, case=False,
                                                       na=False)]
    if obj.empty:
        print(f"    no player matching '{OBJ_NAME}' - skipping OBJ exports")
        return None, None, None
    obj_id = int(obj["nflId"].iloc[0])
    obj_name = obj["displayName"].iloc[0]

    ctx = plays[["gameId", "playId", "week", "quarter", "down", "yardsToGo",
                 "playDescription"]]
    tgt_name = rec_target_names(rec)      # gameId, playId -> target_name

    obj_plays = (
        rec[(rec["nflId"] == obj_id) & rec["invisible"]]
        .merge(ctx, on=["gameId", "playId"], how="left", suffixes=("", "_ctx"))
        .merge(tgt_name, on=["gameId", "playId"], how="left")
    )
    # prefer the play-level yardsToGo from ctx (identical, but keep one column)
    obj_plays = obj_plays.sort_values("separation", ascending=False)

    cols = ["gameId", "playId", "week", "quarter", "down", "yardsToGo",
            "separation", "depth", "target_name", "target_separation",
            "playDescription"]
    obj_out = obj_plays[cols].reset_index(drop=True)
    print(f"    {obj_name}: {len(obj_out)} invisible plays")

    # throw-frame positions for the single most-open OBJ play
    diagram = None
    if len(obj_out):
        top = obj_out.iloc[0]
        gkey, pkey = int(top["gameId"]), int(top["playId"])
        frame = throw[(throw["gameId"] == gkey) & (throw["playId"] == pkey)].copy()
        # tag side for coloring: offense = possessionTeam, else defense/ball
        poss = frame["possessionTeam"].iloc[0] if len(frame) else None
        frame["side"] = np.where(
            frame["is_ball"], "ball",
            np.where(frame["team"] == poss, "offense", "defense"),
        )
        tnm = tgt_name[(tgt_name["gameId"] == gkey) &
                       (tgt_name["playId"] == pkey)]
        target_nflId = None
        if len(tnm):
            tmatch = players[players["displayName"] ==
                             tnm["target_name"].iloc[0]]
            if len(tmatch):
                target_nflId = int(tmatch["nflId"].iloc[0])
        frame["is_obj"] = frame["nflId"] == obj_id
        frame["is_target"] = frame["nflId"] == target_nflId
        keep = ["gameId", "playId", "nflId", "displayName", "officialPosition",
                "team", "side", "x", "y", "los_x", "yardsToGo",
                "is_obj", "is_target", "pff_role"]
        diagram = frame[keep].reset_index(drop=True)
        print(f"    most-open play: game {gkey} play {pkey}, "
              f"OBJ separation {top['separation']:.1f} yd")

    return obj_out, diagram, obj_name


def rec_target_names(rec):
    """Return gameId, playId -> target_name using the targeted route row."""
    t = rec[rec["is_target"]][["gameId", "playId", "displayName"]].copy()
    t = t.drop_duplicates(["gameId", "playId"]).rename(
        columns={"displayName": "target_name"})
    return t


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plays, players, pff, passers = load_static()
    valid_keys = set(map(tuple, plays[["gameId", "playId"]].to_numpy()))

    frames = load_tracking_frames(valid_keys)
    frames = normalize_orientation(frames)
    frames = enrich(frames, pff, players, plays)

    arrival = frames[frames["frame_kind"] == "arrival"].copy()
    window = frames[frames["frame_kind"] == "window"].copy()
    throw = window[window["frame_offset"] == 0].copy()

    targets = identify_targets(arrival, window, plays, players)
    validate_targets(targets, arrival, plays, n=20)

    rec = separation_over_window(window)
    rec = flag_invisible(rec, targets)

    per_player, qualified, cov = aggregate(rec)
    clean = rec[rec["clean_pocket"]].copy()
    per_player_clean, qualified_clean, _ = aggregate(clean)
    qb = qb_missed_open(rec, targets, passers)
    obj_out, obj_diagram, obj_name = obj_exports(rec, throw, players, plays)
    sens = sensitivity_obj(rec)

    # ---- save outputs ----
    route_cols = [
        "gameId", "playId", "nflId", "displayName", "officialPosition",
        "separation", "min_sep_window", "sustained_open", "target_separation",
        "depth", "qb_dist", "yardsToGo", "past_sticks", "is_target",
        "invisible", "costly_miss", "clean_pocket", "pff_blockType",
        "pff_passCoverageType",
    ]
    rate_cols = [
        "nflId", "displayName", "officialPosition", "routes", "open_routes",
        "open_rate", "ignored_open_count", "ignored_when_open",
        "invisible", "invisible_rate", "invisible_rate_ci_low",
        "invisible_rate_ci_high", "costly_miss", "avg_depth",
        "past_sticks_invisible_count",
    ]
    rec[route_cols].to_csv(os.path.join(OUT_DIR, "receiver_routes.csv"), index=False)
    targets.to_csv(os.path.join(OUT_DIR, "play_targets.csv"), index=False)
    per_player.to_csv(os.path.join(OUT_DIR, "player_invisible_all.csv"), index=False)
    qualified[rate_cols].to_csv(
        os.path.join(OUT_DIR, "player_invisible_rate.csv"), index=False)
    qualified_clean[rate_cols].to_csv(
        os.path.join(OUT_DIR, "player_invisible_rate_clean_pocket.csv"),
        index=False)
    cov.to_csv(os.path.join(OUT_DIR, "player_invisible_by_coverage.csv"), index=False)
    qb.to_csv(os.path.join(OUT_DIR, "qb_missed_open_rate.csv"), index=False)
    sens.to_csv(os.path.join(OUT_DIR, "sensitivity_obj.csv"), index=False)
    if obj_out is not None:
        obj_out.to_csv(os.path.join(OUT_DIR, "obj_invisible_plays.csv"), index=False)
    if obj_diagram is not None:
        obj_diagram.to_csv(os.path.join(OUT_DIR, "obj_top_play_frame.csv"),
                           index=False)

    # ---- prints ----
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)

    def _fmt(df):
        d = df.head(15).copy()
        d["invisible_rate"] = (d["invisible_rate"] * 100).round(1)
        d["ci"] = ("[" + (d["invisible_rate_ci_low"] * 100).round(1).astype(str)
                   + ", " + (d["invisible_rate_ci_high"] * 100).round(1).astype(str)
                   + "]")
        d["open_rate"] = (d["open_rate"] * 100).round(1)
        return d[["displayName", "officialPosition", "routes", "invisible",
                  "invisible_rate", "ci", "open_rate", "costly_miss"]]

    print("\n" + "=" * 86)
    print(f"TOP 15 INVISIBLE RECEIVERS - ALL PLAYS (min {MIN_ROUTES} routes)")
    print("rate columns are %; ci = Wilson 95% interval; open_rate = % routes sustained-open")
    print("=" * 86)
    print(_fmt(qualified).to_string(index=False))

    print("\n" + "=" * 86)
    print(f"TOP 15 INVISIBLE RECEIVERS - CLEAN POCKET ONLY (min {MIN_ROUTES} routes)")
    print("=" * 86)
    print(_fmt(qualified_clean).to_string(index=False))

    print("\n" + "=" * 86)
    print(f"TOP 10 QBs BY MISSED-OPEN RATE (min {MIN_QB_PLAYS} plays)")
    print("=" * 86)
    qtop = qb.head(10).copy()
    qtop["qb_missed_open_rate"] = qtop["qb_missed_open_rate"].round(3)
    print(qtop[["passer_name", "plays", "invisible_routes",
                "qb_missed_open_rate"]].to_string(index=False))

    # ---- chip-exclusion impact ----
    print("\n" + "=" * 86)
    print("CHIP (CH) EXCLUSION IMPACT")
    print("=" * 86)
    print(f"  invisible flags before chip exclusion: "
          f"{rec.attrs.get('invisible_before_chip', 'n/a'):,}")
    print(f"  flags removed by excluding CH routes  : "
          f"{rec.attrs.get('chip_removed', 'n/a'):,}")
    print(f"  invisible flags after exclusion       : "
          f"{int(rec['invisible'].sum()):,}")

    # OBJ rank in the qualified (all-plays) table, by invisible_rate
    obj_rank = None
    rank_tbl = qualified.reset_index(drop=True)
    hit = rank_tbl.index[rank_tbl["displayName"].str.contains(
        OBJ_NAME, case=False, na=False)]
    if len(hit):
        obj_rank = int(hit[0]) + 1

    # ---- OBJ spotlight ----
    print("\n" + "=" * 86)
    print("ODELL BECKHAM JR. - SPOTLIGHT")
    print("=" * 86)
    objrow = per_player[per_player["displayName"].str.contains(
        OBJ_NAME, case=False, na=False)]
    if len(objrow):
        o = objrow.iloc[0]
        if obj_rank is not None:
            print(f"  rank among qualified (all plays): "
                  f"{obj_rank} of {len(qualified)}")
        objc = per_player_clean[per_player_clean["displayName"].str.contains(
            OBJ_NAME, case=False, na=False)]
        print(f"  routes (all)          : {int(o['routes'])}")
        print(f"  invisible (all)       : {int(o['invisible'])}  "
              f"rate {o['invisible_rate']:.1%}  "
              f"CI [{o['invisible_rate_ci_low']:.1%}, "
              f"{o['invisible_rate_ci_high']:.1%}]")
        if len(objc):
            oc = objc.iloc[0]
            print(f"  invisible (clean pkt) : {int(oc['invisible'])} of "
                  f"{int(oc['routes'])}  rate {oc['invisible_rate']:.1%}")
        print(f"  open_rate             : {o['open_rate']:.1%}")
        print(f"  ignored_when_open     : {o['ignored_when_open']:.1%}")
        print(f"  costly misses         : {int(o['costly_miss'])}")
        print("  sensitivity (rank / rate):")
        for _, r in sens.iterrows():
            print(f"    sep>={r['sep_threshold']}:  rank {r['obj_rank']}"
                  f"/{r['n_qualified']}  rate {r['obj_invisible_rate']:.1%}")

    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
