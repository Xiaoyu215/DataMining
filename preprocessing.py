"""
preprocessing.py — Data preparation for the Yambda streaming analysis.

The notebook uses these builders to keep analysis separate from wrangling.

RQ1 — per-user feature tables
    load_or_build_features(...) -> (listens, sess, user_features)
        Downloads the 50M-event Yambda parquet, sessionizes, and builds the
        per-user feature table used by RQ1 clustering. Results are cached to
        ./data/ so subsequent runs are seconds, not minutes.

    compute_features_for_period(listen_df) -> user_features
        The same feature pipeline applied to an arbitrary subset of listens
        (e.g. first half of the observation window, or a single week). Used
        by RQ1 temporal stability and by the RQ3 weekly panel.

    auto_archetype_map(labels, feats_df, feat_cols) -> dict
        Rule-based label → archetype-name mapping so "Active samplers"
        always refers to the same behavioral signature across reruns.

RQ2 — anomaly signals
    build_gap_features(listens) -> DataFrame
        Within-session inter-event gap statistics (gap_cv, gap_entropy, ...).

    build_item_concentration(data_path) -> DataFrame
        Per-user top-1 / top-5 item share, via DuckDB on the raw parquet.

    build_event_mix(data_path) -> DataFrame
        Per-user rate_listen / rate_feedback, via DuckDB on the raw parquet.

    build_anomaly_table(user_features, listens, data_path) -> DataFrame
        End-to-end: runs the three builders above and merges them with the
        RQ1 user_features + km_label into a single anomaly_table used by RQ2.

RQ3 — weekly panel + coordination
    build_weekly_feature_tables(listens, n_weeks) -> (weeks, week_feats)
        Picks the N busiest weeks and computes per-user features on each.

    assign_weekly_archetypes(user_features, week_feats, feature_cols)
        -> (panel, centroid_matrix, arch_names, rq1_scaler)
        Fixes the RQ1 centroids and assigns each (user, week) to its nearest
        archetype in RQ1's standardized feature space.

    score_weekly_anomalies(panel, week_feats, weeks, arch_names,
                           anomaly_cols, flag_percentile=95, seed=42) -> panel
        Runs a per-(week, cluster) Isolation Forest and writes anomaly_score
        + is_flagged back into the panel.

    build_coordination_events(listens, weeks) -> DataFrame
        Per-event popularity-adjusted coordination score (RQ3 Phase C).

    build_repeated_peers(ev, pair_min_overlap=3, unusual_cell_min=5) -> DataFrame
        Per-user count of distinct peers they co-occur with in multiple
        unusual cells.

Typical notebook usage
----------------------
    from preprocessing import (
        load_or_build_features,
        compute_features_for_period, auto_archetype_map,
        build_gap_features, build_anomaly_table,
        build_weekly_feature_tables, assign_weekly_archetypes,
        score_weekly_anomalies,
        build_coordination_events, build_repeated_peers,
        DEFAULT_DATA_URL,
    )

    # RQ1
    listens, sess, user_features = load_or_build_features()

    # RQ2 (after clustering has added km_label + archetype to user_features)
    anomaly_table = build_anomaly_table(user_features, listens)

    # RQ3
    weeks, week_feats = build_weekly_feature_tables(listens)
    panel, centroids, arch_names, scaler = assign_weekly_archetypes(
        user_features, week_feats, FEATURE_COLS,
    )
    panel = score_weekly_anomalies(panel, week_feats, weeks, arch_names, anomaly_cols)
    ev = build_coordination_events(listens, weeks)
    repeated_peers = build_repeated_peers(ev)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Configuration — module-level constants (all heuristics live here)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_URL = "hf://datasets/yandex/yambda/flat/50m/multi_event.parquet"
CACHE_DIR = Path("./data")

# RQ1 feature-engineering heuristics
SESSION_GAP_SECONDS = 1800           # 30-minute idle → new session
SKIP_THRESHOLD_PCT = 20              # played_ratio_pct < 20 → skip
FULL_LISTEN_THRESHOLD_PCT = 95       # played_ratio_pct ≥ 95 → full listen
VERY_SHORT_PLAY_SECONDS = 5          # played_seconds < 5 → very-short play
MIN_LISTENS_PER_USER = 5             # users below this are too noisy to cluster

# RQ2 anomaly-signal heuristics
GAP_WITHIN_SESSION_MAX = 1800        # restrict gap analysis to within-session
GAP_HIST_BINS = 50                   # bins for gap-entropy histogram

# RQ3 weekly-panel heuristics
SECONDS_PER_WEEK = 86400 * 7
DEFAULT_N_PANEL_WEEKS = 4            # use the N busiest weeks
MIN_USERS_PER_WEEK = 500             # skip weeks too sparse to cluster
DEFAULT_FLAG_PERCENTILE = 95         # top 5% within cluster-week = flagged

# RQ3 coordination heuristics
HOUR_BIN_SECONDS = 3600
DEFAULT_PAIR_MIN_OVERLAP = 3         # peer pair must share ≥ this many unusual cells
DEFAULT_UNUSUAL_CELL_MIN = 5         # (hr, item) cell must have ≥ this many users

SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# RQ1 — Per-user feature pipeline with parquet caching
# ════════════════════════════════════════════════════════════════════════════

def load_or_build_features(
    data_path: str = DEFAULT_DATA_URL,
    cache_dir: str | Path = CACHE_DIR,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (listens, sess, user_features), using the parquet cache if available."""
    cache_dir = Path(cache_dir)
    listens_path = cache_dir / "listens.parquet"
    sess_path = cache_dir / "sess.parquet"
    user_features_path = cache_dir / "user_features.parquet"
    cache_exists = all(p.exists() for p in (listens_path, sess_path, user_features_path))

    if cache_exists and not force_rebuild:
        if verbose:
            print(f"Loading cached features from {cache_dir}/ ...")
        listens = pd.read_parquet(listens_path)
        sess = pd.read_parquet(sess_path)
        user_features = pd.read_parquet(user_features_path)
        if verbose:
            print(f"  listens       : {len(listens):>10,} rows")
            print(f"  sess          : {len(sess):>10,} rows")
            print(f"  user_features : {len(user_features):>10,} rows")
        return listens, sess, user_features

    if verbose:
        if force_rebuild:
            print("force_rebuild=True → rebuilding from source...")
        else:
            print(f"No cache found at {cache_dir}/ → building from source...")
            print("(This takes a few minutes on the full 50M-event dataset.)")

    cache_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    user_activity = _build_user_activity(con, data_path, verbose)
    listens = _build_listens(con, data_path, verbose)
    sess, user_from_sess = _build_sessions(con, listens, verbose)
    entropy = _build_entropy(listens, verbose)
    user_features = _merge_user_features(
        user_activity, user_from_sess, entropy, verbose
    )

    if verbose:
        print(f"\nWriting cache to {cache_dir}/ ...")
    listens.to_parquet(listens_path, index=False)
    sess.to_parquet(sess_path, index=False)
    user_features.to_parquet(user_features_path, index=False)
    if verbose:
        print("Done.")
    return listens, sess, user_features


def _build_user_activity(con, data_path: str, verbose: bool) -> pd.DataFrame:
    if verbose:
        print("  [1/4] user activity + feedback from all events...")
    user_activity = con.execute(f"""
        SELECT
            uid,
            COUNT(*)                                               AS n_events,
            COUNT(DISTINCT item_id)                                AS n_items,
            AVG(is_organic)                                        AS organic_share,
            (MAX(timestamp) - MIN(timestamp)) * 5.0 / 86400.0      AS t_span_days,
            SUM(CASE WHEN event_type='listen'    THEN 1 ELSE 0 END) AS n_listens,
            SUM(CASE WHEN event_type='like'      THEN 1 ELSE 0 END) AS n_likes,
            SUM(CASE WHEN event_type='unlike'    THEN 1 ELSE 0 END) AS n_unlikes,
            SUM(CASE WHEN event_type='dislike'   THEN 1 ELSE 0 END) AS n_dislikes,
            SUM(CASE WHEN event_type='undislike' THEN 1 ELSE 0 END) AS n_undislikes
        FROM read_parquet('{data_path}')
        GROUP BY uid
    """).df()

    min_span = 1.0 / 86400.0
    span = user_activity["t_span_days"].clip(lower=min_span)
    user_activity["events_per_day"]  = user_activity["n_events"]  / span
    user_activity["listens_per_day"] = user_activity["n_listens"] / span
    for feedback in ("like", "unlike", "dislike", "undislike"):
        user_activity[f"rate_{feedback}"] = (
            user_activity[f"n_{feedback}s"] / user_activity["n_events"]
        )
    if verbose:
        print(f"        {len(user_activity):,} users")
    return user_activity


def _build_listens(con, data_path: str, verbose: bool) -> pd.DataFrame:
    if verbose:
        print("  [2/4] loading listens + sessionizing...")
    listens = con.execute(f"""
        SELECT
            uid,
            timestamp * 5                                  AS t_sec,
            item_id,
            is_organic,
            LEAST(GREATEST(played_ratio_pct, 0), 100)     AS played_clip,
            track_length_seconds
        FROM read_parquet('{data_path}')
        WHERE event_type = 'listen'
        ORDER BY uid, timestamp
    """).df()
    if verbose:
        print(f"        {len(listens):,} listen events")

    listens["skip"] = (listens["played_clip"] < SKIP_THRESHOLD_PCT).astype("int8")
    listens["full_listen"] = (listens["played_clip"] >= FULL_LISTEN_THRESHOLD_PCT).astype("int8")
    listens["played_seconds"] = (
        listens["track_length_seconds"].fillna(0) * listens["played_clip"] / 100.0
    )
    listens["very_short"] = (listens["played_seconds"] < VERY_SHORT_PLAY_SECONDS).astype("int8")

    listens = listens.sort_values(["uid", "t_sec"]).reset_index(drop=True)
    gap = listens.groupby("uid")["t_sec"].diff().fillna(0)
    listens["new_sess"] = (gap > SESSION_GAP_SECONDS).astype("int8")
    listens["session_id"] = listens.groupby("uid")["new_sess"].cumsum()
    listens["prev_item"] = listens.groupby(["uid", "session_id"])["item_id"].shift(1)
    listens["imm_repeat"] = (listens["item_id"] == listens["prev_item"]).astype("int8")
    listens = listens.drop(columns=["new_sess", "prev_item"])
    return listens


def _build_sessions(con, listens: pd.DataFrame, verbose: bool):
    if verbose:
        print("  [3/4] session-level aggregates...")
    con.register("listens_tbl", listens)
    sess = con.execute("""
        SELECT
            uid, session_id,
            COUNT(*)                                     AS n_listens,
            COUNT(DISTINCT item_id)                      AS n_unique,
            AVG(skip::DOUBLE)                            AS skip_rate,
            AVG(full_listen::DOUBLE)                     AS full_rate,
            AVG(imm_repeat::DOUBLE)                      AS imm_repeat_rate,
            AVG(played_clip)                             AS played_ratio_mean,
            AVG(very_short::DOUBLE)                      AS very_short_play_share,
            QUANTILE_CONT(played_clip, 0.75)
              - QUANTILE_CONT(played_clip, 0.25)         AS played_ratio_iqr
        FROM listens_tbl
        GROUP BY uid, session_id
    """).df()

    sess["has_repeat"] = (sess["n_unique"] < sess["n_listens"]).astype("int8")
    sess["repeat_event_rate"] = (
        (sess["n_listens"] - sess["n_unique"]) / sess["n_listens"].clip(lower=1)
    )

    user_from_sess = sess.groupby("uid").agg(
        n_sessions=("n_listens", "size"),
        listens_per_session_med=("n_listens", "median"),
        n_unique_items_per_session_med=("n_unique", "median"),
        skip_rate_med=("skip_rate", "median"),
        full_rate_med=("full_rate", "median"),
        repeat_session_rate=("has_repeat", "median"),
        repeat_event_rate_med=("repeat_event_rate", "median"),
        imm_repeat_rate_med=("imm_repeat_rate", "median"),
        played_ratio_med=("played_ratio_mean", "median"),
        played_ratio_iqr=("played_ratio_iqr", "median"),
        very_short_play_share=("very_short_play_share", "median"),
    ).reset_index()
    if verbose:
        print(f"        {len(sess):,} sessions across {len(user_from_sess):,} users")
    return sess, user_from_sess


def _build_entropy(listens: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    if verbose:
        print("  [4/4] item diversity (entropy) per user...")
    counts = listens.groupby(["uid", "item_id"]).size().reset_index(name="cnt")
    totals = counts.groupby("uid")["cnt"].transform("sum")
    counts["p"] = counts["cnt"] / totals
    entropy = (
        counts.groupby("uid")
        .apply(lambda g: -np.sum(g["p"] * np.log2(g["p"] + 1e-12)))
        .reset_index()
    )
    entropy.columns = ["uid", "item_entropy"]
    return entropy


def _merge_user_features(user_activity, user_from_sess, entropy, verbose):
    user_features = (
        user_activity
        .merge(user_from_sess, on="uid", how="inner")
        .merge(entropy, on="uid", how="left")
    )
    user_features["catalog_breadth"] = (
        user_features["n_items"] / user_features["n_listens"].clip(lower=1)
    )

    n_before = len(user_features)
    user_features = user_features[user_features["n_listens"] >= MIN_LISTENS_PER_USER].copy()
    if verbose:
        print(f"        dropped {n_before - len(user_features):,} users with < {MIN_LISTENS_PER_USER} listens")
    user_features = user_features.fillna(0).sort_values("uid").reset_index(drop=True)
    if verbose:
        print(f"        final user_features: {user_features.shape}")
    return user_features


# ════════════════════════════════════════════════════════════════════════════
# Shared helpers — used by RQ1 temporal stability AND RQ3 weekly panel
# ════════════════════════════════════════════════════════════════════════════

def compute_features_for_period(listen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature pipeline to an arbitrary subset of listens.

    Unlike ``load_or_build_features`` this works on an already-loaded slice
    of the ``listens`` DataFrame (e.g. ``listens[listens.week == 3]``) and
    doesn't touch the raw parquet. Used by RQ1 temporal stability and the
    RQ3 weekly panel.
    """
    if len(listen_df) == 0:
        return pd.DataFrame()

    listen_df = listen_df.copy()
    listen_df = listen_df.sort_values(["uid", "t_sec"])
    gap = listen_df.groupby("uid")["t_sec"].diff().fillna(0)
    listen_df["new_sess"] = (gap > SESSION_GAP_SECONDS).astype("int8")
    listen_df["session_id"] = listen_df.groupby("uid")["new_sess"].cumsum()
    listen_df["prev_item"] = listen_df.groupby(["uid", "session_id"])["item_id"].shift(1)
    listen_df["imm_repeat"] = (listen_df["item_id"] == listen_df["prev_item"]).astype("int8")
    # Note: original notebook used <= 10 as the very-short threshold here,
    # distinct from the full pipeline's 5-second cutoff. Preserved as-is.
    listen_df["very_short"] = (listen_df["played_clip"] <= 10).astype("int8")

    sess_p = listen_df.groupby(["uid", "session_id"]).agg(
        n_listens=("item_id", "size"),
        n_unique=("item_id", "nunique"),
        imm_repeat_rate=("imm_repeat", "mean"),
        played_ratio_mean=("played_clip", "mean"),
        played_ratio_std=("played_clip", "std"),
        very_short_play_share=("very_short", "mean"),
    ).reset_index()
    sess_p["has_repeat"] = (sess_p["n_unique"] < sess_p["n_listens"]).astype("int8")
    sess_p["repeat_event_rate"] = (
        (sess_p["n_listens"] - sess_p["n_unique"]) / sess_p["n_listens"].clip(lower=1)
    )

    uf = sess_p.groupby("uid").agg(
        n_sessions=("n_listens", "size"),
        listens_per_session_med=("n_listens", "median"),
        n_unique_items_per_session_med=("n_unique", "median"),
        repeat_session_rate=("has_repeat", "mean"),
        repeat_event_rate_med=("repeat_event_rate", "median"),
        imm_repeat_rate_med=("imm_repeat_rate", "median"),
        played_ratio_mean=("played_ratio_mean", "mean"),
        played_ratio_iqr=("played_ratio_std", "mean"),   # proxy via std
        very_short_play_share=("very_short_play_share", "mean"),
    ).reset_index()

    ic = listen_df.groupby(["uid", "item_id"]).size().reset_index(name="cnt")
    tot = ic.groupby("uid")["cnt"].transform("sum")
    ic["p"] = ic["cnt"] / tot
    ent = (
        ic.groupby("uid")
        .apply(lambda g: -np.sum(g["p"] * np.log2(g["p"] + 1e-12)))
        .reset_index()
    )
    ent.columns = ["uid", "item_entropy"]

    n_items = listen_df.groupby("uid")["item_id"].nunique().reset_index(name="n_items")
    n_list = listen_df.groupby("uid").size().reset_index(name="n_listens_p")
    t_span = (
        listen_df.groupby("uid")["t_sec"]
        .agg(lambda x: (x.max() - x.min()) / 86400)
        .reset_index(name="t_span_days")
    )
    org = listen_df.groupby("uid")["is_organic"].mean().reset_index(name="organic_share")

    uf = (
        uf.merge(ent, on="uid", how="left")
        .merge(n_items, on="uid", how="left")
        .merge(n_list, on="uid", how="left")
        .merge(t_span, on="uid", how="left")
        .merge(org, on="uid", how="left")
    )

    uf["catalog_breadth"] = uf["n_items"] / uf["n_listens_p"].clip(lower=1)
    uf["activity_level"] = np.log1p(
        uf["n_listens_p"] / uf["t_span_days"].clip(lower=1 / 86400)
    )
    return uf.fillna(0)


def auto_archetype_map(labels, feats_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """
    Rule-based label → archetype-name mapping so archetype names are stable
    across reruns (cluster IDs are arbitrary integers, but the *named*
    archetypes always point to the same behavioral signature).

    Rules:
        highest repeat_session_rate       → "Active samplers"
        highest item_entropy              → "Casual explorers"
        remainder                         → "Lean-back loyal listeners"
    """
    tmp = feats_df[feat_cols].copy()
    tmp["_label"] = labels
    replay_c = tmp.groupby("_label")["repeat_session_rate"].mean().idxmax()
    explorer_c = tmp.groupby("_label")["item_entropy"].mean().idxmax()
    mapping = {}
    for k in tmp["_label"].unique():
        if k == replay_c:
            mapping[k] = "Active samplers"
        elif k == explorer_c:
            mapping[k] = "Casual explorers"
        else:
            mapping[k] = "Lean-back loyal listeners"
    return mapping


# ════════════════════════════════════════════════════════════════════════════
# RQ2 — Anomaly signal engineering
# ════════════════════════════════════════════════════════════════════════════

def build_gap_features(listens: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Per-user statistics on within-session inter-event gaps.

    Bots tend to emit events on a metronome; humans have noisy timing. The
    key column ``gap_cv`` (std / mean) is low for bots, high for humans.

    Returns columns:
        uid, gap_n, gap_med, gap_mean, gap_std, gap_iqr,
        pct_gap_5s, pct_gap_10s, pct_gap_15s, gap_entropy,
        gap_cv, pct_exact_bin
    """
    if verbose:
        print("Computing inter-event gap features...")

    listens_sorted = listens.sort_values(["uid", "t_sec"])
    gap_series = listens_sorted.groupby("uid")["t_sec"].diff().dropna()
    gap_df = pd.DataFrame({
        "uid": listens_sorted.loc[gap_series.index, "uid"].values,
        "gap": gap_series.values,
    })
    gap_df = gap_df[gap_df["gap"] <= GAP_WITHIN_SESSION_MAX].copy()

    def gap_entropy(x, bins=GAP_HIST_BINS):
        counts, _ = np.histogram(x, bins=bins, range=(0, GAP_WITHIN_SESSION_MAX))
        p = counts / (counts.sum() + 1e-12)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    gap_stats = gap_df.groupby("uid").agg(
        gap_n=("gap", "size"),
        gap_med=("gap", "median"),
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
        gap_iqr=("gap", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        pct_gap_5s=("gap", lambda x: (x == 5).mean()),
        pct_gap_10s=("gap", lambda x: (x == 10).mean()),
        pct_gap_15s=("gap", lambda x: (x == 15).mean()),
        gap_entropy=("gap", gap_entropy),
    ).reset_index()

    gap_stats["gap_cv"] = gap_stats["gap_std"] / (gap_stats["gap_mean"] + 1e-9)
    gap_stats["pct_exact_bin"] = (
        gap_stats["pct_gap_5s"] + gap_stats["pct_gap_10s"] + gap_stats["pct_gap_15s"]
    )

    if verbose:
        print(f"  gap features for {len(gap_stats):,} users")
    return gap_stats


def build_item_concentration(
    data_path: str = DEFAULT_DATA_URL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per-user top-1 and top-5 item share — the fraction of listens going to
    the user's most-played track(s). A bot boosting a specific track has
    very high ``top1_item_share``.

    Uses DuckDB directly on the raw parquet for efficiency.
    Returns: uid, top1_item_share, top5_item_share
    """
    if verbose:
        print("Computing per-user item concentration...")

    con = duckdb.connect()
    item_conc = con.execute(f"""
        WITH item_counts AS (
            SELECT uid, item_id, COUNT(*) AS cnt
            FROM read_parquet('{data_path}')
            WHERE event_type = 'listen'
            GROUP BY uid, item_id
        ),
        user_totals AS (
            SELECT uid, SUM(cnt) AS total_listens
            FROM item_counts
            GROUP BY uid
        ),
        ranked AS (
            SELECT ic.uid, ic.cnt, ut.total_listens,
                   ROW_NUMBER() OVER (PARTITION BY ic.uid ORDER BY ic.cnt DESC) AS rnk
            FROM item_counts ic JOIN user_totals ut ON ic.uid = ut.uid
        )
        SELECT
            uid,
            SUM(CASE WHEN rnk = 1 THEN cnt ELSE 0 END)::DOUBLE / MAX(total_listens) AS top1_item_share,
            SUM(CASE WHEN rnk <= 5 THEN cnt ELSE 0 END)::DOUBLE / MAX(total_listens) AS top5_item_share
        FROM ranked
        GROUP BY uid
    """).df()

    if verbose:
        print(f"  {len(item_conc):,} users")
    return item_conc


def build_event_mix(
    data_path: str = DEFAULT_DATA_URL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per-user event-type mix (across all events, not just listens).

    ``rate_listen`` close to 1.0 means the user emits almost no feedback —
    bot-like. Real users intermix likes/dislikes into their event stream.

    Returns: uid, n_events, rate_listen, rate_feedback
    """
    if verbose:
        print("Computing per-user event mix...")

    con = duckdb.connect()
    event_mix = con.execute(f"""
        SELECT
            uid,
            COUNT(*)                                                            AS n_events,
            SUM(CASE WHEN event_type='listen' THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*)                                                      AS rate_listen,
            (SUM(CASE WHEN event_type='like'    THEN 1 ELSE 0 END)
           + SUM(CASE WHEN event_type='dislike' THEN 1 ELSE 0 END))::DOUBLE
                / COUNT(*)                                                      AS rate_feedback
        FROM read_parquet('{data_path}')
        GROUP BY uid
    """).df()

    if verbose:
        print(f"  {len(event_mix):,} users")
    return event_mix


def build_anomaly_table(
    user_features: pd.DataFrame,
    listens: pd.DataFrame,
    data_path: str = DEFAULT_DATA_URL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    End-to-end RQ2 feature builder.

    Runs the three signal builders above and merges them with the RQ1
    ``user_features`` into a single anomaly-ready table.

    Expects ``user_features`` to already have ``km_label`` assigned (add
    it after running K-Means). The archetype name is auto-derived from
    km_label using the same rule as ``auto_archetype_map``, so the result
    doesn't depend on which integer label was assigned to which cluster.
    """
    assert "km_label" in user_features.columns, (
        "user_features must have a 'km_label' column from clustering before "
        "calling build_anomaly_table(...)"
    )

    gap_stats = build_gap_features(listens, verbose=verbose)
    item_conc = build_item_concentration(data_path, verbose=verbose)
    event_mix = build_event_mix(data_path, verbose=verbose)

    # Re-derive archetype from km_label at build time so the table is robust
    # to label permutation between runs.
    replay_cluster = user_features.groupby("km_label")["repeat_session_rate"].mean().idxmax()
    explorer_cluster = user_features.groupby("km_label")["item_entropy"].mean().idxmax()
    archetype_map = {}
    for k in user_features["km_label"].unique():
        if k == replay_cluster:
            archetype_map[k] = "Active samplers"
        elif k == explorer_cluster:
            archetype_map[k] = "Casual explorers"
        else:
            archetype_map[k] = "Lean-back loyal listeners"
    if verbose:
        print(f"Auto-detected archetype map: {archetype_map}")

    anomaly_table = (
        user_features[[
            "uid", "events_per_day", "listens_per_day", "organic_share",
            "repeat_session_rate", "repeat_event_rate_med", "imm_repeat_rate_med",
            "km_label",
        ]]
        .assign(archetype=lambda d: d["km_label"].map(archetype_map))
        .merge(
            gap_stats[["uid", "gap_cv", "gap_entropy", "pct_exact_bin", "gap_med"]],
            on="uid", how="left",
        )
        .merge(item_conc, on="uid", how="left")
        .merge(event_mix, on="uid", how="left")
    )
    anomaly_table = anomaly_table.fillna(anomaly_table.median(numeric_only=True))
    if verbose:
        print(f"anomaly_table: {anomaly_table.shape}")
    return anomaly_table


# ════════════════════════════════════════════════════════════════════════════
# RQ3 — Weekly panel + centroid-based archetype assignment
# ════════════════════════════════════════════════════════════════════════════

def build_weekly_feature_tables(
    listens: pd.DataFrame,
    n_weeks: int = DEFAULT_N_PANEL_WEEKS,
    min_users_per_week: int = MIN_USERS_PER_WEEK,
    verbose: bool = True,
) -> tuple[list[int], dict[int, pd.DataFrame]]:
    """
    Pick ``n_weeks`` evenly-spaced weeks and run the feature pipeline on each.

    Selection rule:
        1. Prefer consecutive weeks (gap = 1). Among all windows of size
           ``n_weeks`` where every week has ≥ ``min_users_per_week`` distinct
           users, pick the one with the highest total listens.
        2. If no consecutive window qualifies, try uniform gaps 2, 3, 4, ...
           until a qualifying window is found. Same "total listens" tiebreaker.
        3. If no uniform-gap window qualifies at all, fall back to the N
           busiest weeks (with a warning), to keep the pipeline runnable.

    Using evenly-spaced weeks matters for the transition analysis: a gap of
    1 between weeks w and w+1 is what makes "consecutive-week flag carryover"
    a meaningful measurement. A gap of 2, 3, ... is still interpretable as
    long as it's uniform across the panel — "do flags survive a 2-week gap?"
    is a valid question, just a weaker one than the consecutive case.

    Returns (weeks, week_feats):
        weeks — sorted list of week indices used
        week_feats — dict mapping week → DataFrame of per-user features,
                     indexed by uid
    """
    if "week" not in listens.columns:
        listens["week"] = (listens["t_sec"] // SECONDS_PER_WEEK).astype(int)

    # Per-week aggregates (cheap proxies used for window selection)
    week_listens = listens["week"].value_counts().sort_index()         # total listens
    week_users = listens.groupby("week")["uid"].nunique().sort_index()  # distinct users
    all_weeks = sorted(week_listens.index.tolist())

    def find_best_window(gap: int) -> list[int] | None:
        """Best window of ``n_weeks`` weeks spaced by ``gap``, or None."""
        best_weeks, best_total = None, -1
        for start in all_weeks:
            window = [start + i * gap for i in range(n_weeks)]
            # all weeks must actually exist in the data
            if not all(w in week_listens.index for w in window):
                continue
            # all weeks must meet the min-users threshold (cheap proxy check)
            if not all(week_users.get(w, 0) >= min_users_per_week for w in window):
                continue
            total = sum(week_listens[w] for w in window)
            if total > best_total:
                best_weeks, best_total = window, total
        return best_weeks

    # 1. Try consecutive first
    weeks = find_best_window(gap=1)
    chosen_gap = 1

    # 2. Fall back to increasing uniform gaps
    if weeks is None:
        # Upper bound on gap — beyond this, no window of n_weeks fits in the data
        max_gap = max(1, (max(all_weeks) - min(all_weeks)) // (n_weeks - 1))
        for g in range(2, max_gap + 1):
            weeks = find_best_window(gap=g)
            if weeks is not None:
                chosen_gap = g
                break

    # 3. Last-resort fallback: top-N busiest, not uniform-gap
    if weeks is None:
        weeks = week_listens.sort_values(ascending=False).head(n_weeks).index.sort_values().tolist()
        chosen_gap = None
        if verbose:
            print("⚠ No uniform-gap window met the min-users threshold.")
            print(f"  Falling back to the {n_weeks} busiest weeks (non-uniform spacing).")

    if verbose:
        if chosen_gap == 1:
            print(f"Panel weeks: {weeks} (consecutive)")
        elif chosen_gap is not None:
            print(f"Panel weeks: {weeks} (uniform gap = {chosen_gap})")
        else:
            print(f"Panel weeks: {weeks}")
        print(f"Listens per week: {[int(week_listens[w]) for w in weeks]}")

    # Compute features for each selected week
    week_feats: dict[int, pd.DataFrame] = {}
    for w in weeks:
        lw = listens[listens["week"] == w]
        fw = compute_features_for_period(lw)
        if len(fw) < min_users_per_week:
            if verbose:
                print(f"  Week {w}: only {len(fw)} users with features — skipping")
            continue
        week_feats[w] = fw.set_index("uid")
        if verbose:
            print(f"  Week {w}: {len(fw):,} users with features")

    return weeks, week_feats


def assign_weekly_archetypes(
    user_features: pd.DataFrame,
    week_feats: dict[int, pd.DataFrame],
    feature_cols: list[str],
    verbose: bool = True,
):
    """
    Assign each (user, week) to the nearest RQ1 archetype centroid.

    We fix the centroids from RQ1 (not re-cluster each week) so that
    "Active samplers" means the same thing in every week.

    Requires ``user_features`` to have an 'archetype' column already
    assigned.

    Returns:
        panel — DataFrame with columns (uid, week, archetype)
        centroid_matrix — np.ndarray, one row per archetype (alphabetical)
        arch_names — sorted archetype names matching centroid_matrix rows
        rq1_scaler — fitted StandardScaler from RQ1 feature space
    """
    from sklearn.preprocessing import StandardScaler

    assert "archetype" in user_features.columns, (
        "user_features must have an 'archetype' column before assigning "
        "weekly archetypes; run the cluster-name mapping first."
    )

    stab_cols = list(feature_cols)
    rq1_scaler = StandardScaler().fit(user_features[stab_cols].fillna(0))

    X_rq1 = rq1_scaler.transform(user_features[stab_cols].fillna(0))
    arch_series = user_features["archetype"].values
    centroids = {
        arch: X_rq1[arch_series == arch].mean(axis=0)
        for arch in np.unique(arch_series)
    }
    arch_names = sorted(centroids)
    centroid_matrix = np.vstack([centroids[a] for a in arch_names])
    if verbose:
        print(f"Centroid shape: {centroid_matrix.shape}")
        print(f"Archetype order: {arch_names}")

    panel_rows = []
    for w, ft in week_feats.items():
        for c in stab_cols:
            if c not in ft.columns:
                ft[c] = 0.0
        Xw = rq1_scaler.transform(ft[stab_cols].fillna(0))
        dists = np.linalg.norm(Xw[:, None, :] - centroid_matrix[None, :, :], axis=2)
        nearest = dists.argmin(axis=1)
        for uid, k in zip(ft.index, nearest):
            panel_rows.append({"uid": int(uid), "week": w, "archetype": arch_names[k]})

    panel = pd.DataFrame(panel_rows)
    if verbose:
        print(f"Panel rows: {len(panel):,}")
    return panel, centroid_matrix, arch_names, rq1_scaler


def score_weekly_anomalies(
    panel: pd.DataFrame,
    week_feats: dict[int, pd.DataFrame],
    weeks: list[int],
    arch_names: list[str],
    anomaly_cols: list[str],
    flag_percentile: int = DEFAULT_FLAG_PERCENTILE,
    min_cluster_week_users: int = 30,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit a per-(week, cluster) Isolation Forest and attach scores + flags to
    the weekly panel.

    This is the central methodological move of RQ3: anomaly is judged
    within the user's cluster-week, so "flagged" always means "extreme
    relative to peers of the same listening type in the same week".

    Returns the panel with two new columns: ``anomaly_score`` and
    ``is_flagged``.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest

    score_records = []
    for w in weeks:
        if w not in week_feats:
            continue
        ft = week_feats[w]
        sub_panel = panel[panel["week"] == w]

        for arch in arch_names:
            uids = sub_panel.loc[sub_panel["archetype"] == arch, "uid"].values
            common = ft.index.intersection(uids)
            if len(common) < min_cluster_week_users:
                continue
            X_a = StandardScaler().fit_transform(
                ft.loc[common, anomaly_cols].fillna(0)
            )
            iso = IsolationForest(
                n_estimators=200, contamination="auto",
                random_state=seed, n_jobs=-1,
            ).fit(X_a)
            scores = -iso.score_samples(X_a)   # higher = more anomalous
            thresh = np.percentile(scores, flag_percentile)
            flags = (scores >= thresh).astype(int)
            for uid, sc, fl in zip(common, scores, flags):
                score_records.append({
                    "uid": int(uid), "week": w, "archetype": arch,
                    "anomaly_score": float(sc), "is_flagged": int(fl),
                })

    scores_df = pd.DataFrame(score_records)
    panel = panel.merge(scores_df.drop(columns=["archetype"]), on=["uid", "week"], how="left")
    panel["is_flagged"] = panel["is_flagged"].fillna(0).astype(int)
    if verbose:
        print(f"Panel with scores: {len(panel):,} rows")
        print(f"Flagged rows:      {panel['is_flagged'].sum():,}")
        print(f"Flag rate:         {panel['is_flagged'].mean() * 100:.1f}%")
    return panel


# ════════════════════════════════════════════════════════════════════════════
# RQ3 Phase C — Network coordination signals
# ════════════════════════════════════════════════════════════════════════════

def build_coordination_events(
    listens: pd.DataFrame,
    weeks: Iterable[int],
    hour_bin_sec: int = HOUR_BIN_SECONDS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per-event popularity-adjusted coordination score.

    For each (hour_bin, item_id) cell we compute observed vs. expected count
    under independence (item popularity × hour traffic / total). Events that
    land in over-populated cells get a higher ``coord_event_score``.

    Real users sometimes land in popular cells by accident; persistent
    coordination shows up as many events in cells that are unexpectedly
    crowded given the baseline item + hour traffic.

    Returns the event-level DataFrame (one row per listen in the panel
    weeks) augmented with ``hr_bin``, ``n_users_in_cell``, ``cell_size``,
    ``item_pop``, ``hr_pop``, ``expected_cell``, ``excess``, ``coord_event_score``.
    """
    if verbose:
        print("Building coordination events...")

    ev = listens[["uid", "t_sec", "item_id"]].copy()
    ev["week"] = (ev["t_sec"] // SECONDS_PER_WEEK).astype(int)
    ev["hr_bin"] = (ev["t_sec"] // hour_bin_sec).astype("int32")
    ev = ev[ev["week"].isin(list(weeks))]

    cell_users = ev.groupby(["hr_bin", "item_id"])["uid"].nunique().rename("n_users_in_cell")
    cell_size = ev.groupby(["hr_bin", "item_id"]).size().rename("cell_size")
    item_pop = ev.groupby("item_id").size().rename("item_pop")
    hr_pop = ev.groupby("hr_bin").size().rename("hr_pop")
    total = len(ev)

    ev = ev.join(cell_users, on=["hr_bin", "item_id"])
    ev = ev.join(cell_size, on=["hr_bin", "item_id"])
    ev = ev.join(item_pop, on="item_id")
    ev = ev.join(hr_pop, on="hr_bin")

    ev["expected_cell"] = ev["item_pop"] * ev["hr_pop"] / total
    ev["excess"] = (ev["cell_size"] - ev["expected_cell"]).clip(lower=0)
    ev["coord_event_score"] = ev["excess"] / np.sqrt(ev["expected_cell"].clip(lower=1))

    if verbose:
        print(f"  Events: {len(ev):,}")
    return ev


def build_repeated_peers(
    ev: pd.DataFrame,
    pair_min_overlap: int = DEFAULT_PAIR_MIN_OVERLAP,
    unusual_cell_min: int = DEFAULT_UNUSUAL_CELL_MIN,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per-user count of distinct peers they co-occur with in multiple unusual
    (over-populated) (hour_bin, item_id) cells.

    "Unusual" = cell has ≥ unusual_cell_min distinct users AND excess > 0
    (more users than independence would predict).

    A high ``n_repeated_peers`` means this user shows up alongside the same
    small set of other users in crowded cells again and again — the hallmark
    of a coordinated cluster.

    Returns: uid, n_repeated_peers
    """
    if verbose:
        print("Computing repeated-peer overlap...")

    unusual = ev[(ev["n_users_in_cell"] >= unusual_cell_min) &
                 (ev["excess"] > ev["expected_cell"])]
    if verbose:
        print(f"  Unusual cell-events: {len(unusual):,}")

    cell_members = (
        unusual.drop_duplicates(["hr_bin", "item_id", "uid"])
        [["hr_bin", "item_id", "uid"]]
    )

    if verbose:
        print("  Computing pairwise co-occurrence on unusual cells...")
    cm = cell_members.merge(cell_members, on=["hr_bin", "item_id"])
    cm = cm[cm["uid_x"] < cm["uid_y"]]
    pair_counts = (
        cm.groupby(["uid_x", "uid_y"]).size().rename("n_shared_cells").reset_index()
    )
    strong_pairs = pair_counts[pair_counts["n_shared_cells"] >= pair_min_overlap]
    if verbose:
        print(f"  Strong pairs (≥{pair_min_overlap} shared unusual cells): {len(strong_pairs):,}")

    left = strong_pairs.groupby("uid_x").size().rename("n_peers_l")
    right = strong_pairs.groupby("uid_y").size().rename("n_peers_r")
    repeated_peers = (
        pd.concat([left, right], axis=1).fillna(0).sum(axis=1)
        .rename("n_repeated_peers").reset_index()
        .rename(columns={"index": "uid"})
    )
    return repeated_peers


# ════════════════════════════════════════════════════════════════════════════
# CLI — pre-build the RQ1 cache from a terminal
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-build the Yambda RQ1 feature cache under ./data/."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_URL,
                        help=f"Source parquet (default: {DEFAULT_DATA_URL})")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR),
                        help=f"Cache directory (default: {CACHE_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the cache exists.")
    args = parser.parse_args()

    load_or_build_features(
        data_path=args.data_path,
        cache_dir=args.cache_dir,
        force_rebuild=args.force,
        verbose=True,
    )
