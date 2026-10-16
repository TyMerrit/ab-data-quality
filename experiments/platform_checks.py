"""Step 0 demo: validate the platform before trusting any experiment on it.

1. Three assignment functions through the hashing checks:
   - the reference sha256 implementation,
   - a classic modulo-bias bug (two hex digits mod 100),
   - a "global salt" bug (every experiment uses the same salt).
2. A/A replay of two analyses on the same historical data:
   - unit-level t-test (aggregate to the randomization unit first),
   - event-level t-test (every session treated as independent).

Run:  python experiments/platform_checks.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _style  # noqa: E402
from abdq.platform import (  # noqa: E402
    aa_replay,
    assign_variants,
    check_hash_determinism,
    check_hash_independence,
    check_hash_uniformity,
    event_level_ttest,
    unit_level_ttest,
)

SPLIT = {"control": 0.5, "treatment": 0.5}


def good(ids, salt):
    return assign_variants(ids, salt, SPLIT)


def modulo_bias(ids, salt):
    """Bucket = first two hex digits (0..255) mod 100. Buckets 0-55 get 3 hashes, 56-99 get 2."""
    b = np.array([int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest()[:2], 16) % 100 for u in ids])
    return np.where(b < 50, "control", "treatment")


def global_salt(ids, salt):
    """Ignores the experiment salt: every experiment splits users the same way."""
    return assign_variants(ids, "global", SPLIT)


def historical_sessions(n_users: int = 4_000, seed: int = 0) -> pd.DataFrame:
    """Session-level revenue with strong per-user heterogeneity (sessions of one user correlate)."""
    rng = np.random.default_rng(seed)
    user_level = rng.lognormal(1.0, 1.0, size=n_users)
    sessions = 1 + rng.poisson(6, size=n_users)
    uid = np.repeat(np.arange(n_users), sessions)
    value = user_level[uid] * rng.lognormal(0, 0.5, size=len(uid))
    return pd.DataFrame({"unit_id": uid, "value": value})


def main() -> None:
    _style.apply()
    rows = []
    for name, fn in [("sha256 reference", good), ("modulo bias bug", modulo_bias), ("global salt bug", global_salt)]:
        for check in (check_hash_uniformity(fn, SPLIT, n_ids=300_000),
                      check_hash_determinism(fn, n_ids=50_000),
                      check_hash_independence(fn, SPLIT, n_ids=200_000)):
            rows.append({"assignment": name, "check": check.name, "status": check.status.value, "message": check.message})
            print(f"[{name:18s}] {check.status.value.upper():5s} {check.name}: {check.message}")
    pd.DataFrame(rows).to_csv(_style.RESULTS / "platform_hash_checks.csv", index=False)

    events = historical_sessions()
    res_unit = aa_replay(events, unit_level_ttest, n_replays=500, name="aa_replay_unit_level")
    res_event = aa_replay(events, event_level_ttest, n_replays=500, name="aa_replay_event_level")
    for r in (res_unit, res_event):
        print(f"{r.check.status.value.upper():5s} {r.check.name}: {r.check.message}")
    pd.DataFrame([
        {"analysis": "unit-level t-test", **{k: r for k, r in zip(["fpr", "ks_p_value"], [res_unit.false_positive_rate, res_unit.ks_p_value])}},
        {"analysis": "event-level t-test", **{k: r for k, r in zip(["fpr", "ks_p_value"], [res_event.false_positive_rate, res_event.ks_p_value])}},
    ]).to_csv(_style.RESULTS / "platform_aa_replay.csv", index=False)

    fig, axes = _style.plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, r, title, color in [
        (axes[0], res_unit, "Unit-level analysis", _style.SERIES[0]),
        (axes[1], res_event, "Session-level analysis", _style.SERIES[1]),
    ]:
        ax.hist(r.p_values, bins=bins, color=color, edgecolor=_style.SURFACE, linewidth=2)
        ax.axhline(len(r.p_values) / 20, color=_style.NEUTRAL, linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("p-value under A/A")
        ax.text(0.98, 0.95, f"p < 0.05 in {r.false_positive_rate:.0%} of replays",
                transform=ax.transAxes, ha="right", va="top", color=_style.TEXT, fontsize=10)
    axes[0].set_ylabel("replays")
    fig.suptitle("500 A/A replays on the same data: p-values should be flat",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=_style.TEXT)
    fig.tight_layout()
    print("saved", _style.save(fig, "aa_replay_pvalues.png"))


if __name__ == "__main__":
    main()
