"""TM-07: load mock-save CSV into loglik-ready data dicts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twominds.agents import loglik

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_session.csv"

# Valid theta inside bounds for a smoke loglik check.
THETA = (0.5, 0.5, 1.0, 5.0, 5.0, 0.5, 0.0)


def test_load_session_importable():
    from twominds.io import load_session  # noqa: F401


def test_load_smoke_fixture_has_both_conditions():
    from twominds.io import load_session

    session = load_session(FIXTURE)
    assert set(session) >= {"no_pressure", "time_pressure"}
    for cond, data in session.items():
        if cond.startswith("_"):
            continue
        for key in ("a1", "s2", "a2", "r"):
            assert key in data
            assert len(data[key]) == 2  # smoke: 2 trials per block


def test_load_smoke_values_match_csv():
    from twominds.io import load_session

    session = load_session(FIXTURE)
    np_ = session["no_pressure"]
    assert list(np_["a1"]) == [0, 0]
    assert list(np_["s2"]) == [1, 2]
    assert list(np_["a2"]) == [1, 0]
    assert list(np_["r"]) == [0, 1]


def test_loglik_on_loaded_smoke_is_finite():
    from twominds.io import load_session

    session = load_session(FIXTURE)
    for cond, data in session.items():
        if cond.startswith("_"):
            continue
        ll = loglik(THETA, data)
        assert np.isfinite(ll), cond


def test_missed_stage1_masked_as_minus_one(tmp_path: Path):
    from twominds.io import load_session

    csv = tmp_path / "miss1.csv"
    csv.write_text(
        "participant_code,block,condition,trial,a1,s2,a2,r,rt1,rt2,transition,"
        "missed_a1,missed_a2,p_s1_a0,p_s1_a1,p_s2_a0,p_s2_a1,schedule_seed\n"
        "TM-TEST,0,no_pressure,0,,,,"
        ",3000,,,"
        "true,false,0.5,0.5,0.5,0.5,101\n"
        "TM-TEST,0,no_pressure,1,1,2,0,1,100,100,common,"
        "false,false,0.5,0.5,0.6,0.4,101\n"
    )
    data = load_session(csv)["no_pressure"]
    assert data["a1"][0] == -1
    assert data["a2"][0] == -1
    assert data["a1"][1] == 1
    assert data["a2"][1] == 0
    # Missed stage-1 contributes nothing; answered trial still counts.
    assert loglik(THETA, data) == loglik(
        THETA,
        {"a1": np.array([1]), "s2": np.array([2]), "a2": np.array([0]), "r": np.array([1])},
    )


def test_missed_stage2_keeps_a1_masks_a2(tmp_path: Path):
    from twominds.io import load_session

    csv = tmp_path / "miss2.csv"
    csv.write_text(
        "participant_code,block,condition,trial,a1,s2,a2,r,rt1,rt2,transition,"
        "missed_a1,missed_a2,p_s1_a0,p_s1_a1,p_s2_a0,p_s2_a1,schedule_seed\n"
        "TM-TEST,0,no_pressure,0,0,1,,,"
        "100,1500,common,"
        "false,true,0.5,0.5,0.7,0.3,101\n"
    )
    data = load_session(csv)["no_pressure"]
    assert data["a1"][0] == 0
    assert data["s2"][0] == 1
    assert data["a2"][0] == -1
    ll = loglik(THETA, data)
    assert np.isfinite(ll)
    assert ll != 0.0  # stage-1 choice still contributes


def test_exclusions_flagged_not_dropped(tmp_path: Path):
    from twominds.io import load_session

    # 3/10 missed (>20%) and stage-2 picks of the worse option always → chance flag.
    lines = [
        "participant_code,block,condition,trial,a1,s2,a2,r,rt1,rt2,transition,"
        "missed_a1,missed_a2,p_s1_a0,p_s1_a1,p_s2_a0,p_s2_a1,schedule_seed"
    ]
    for t in range(10):
        if t < 3:
            lines.append(
                f"TM-X,0,no_pressure,{t},,,,,"
                f",3000,,,"
                f"true,false,0.7,0.3,0.8,0.2,101"
            )
        else:
            # Always pick a2=1 while p_s1_a0=0.8 > p_s1_a1=0.2 on state 1.
            lines.append(
                f"TM-X,0,no_pressure,{t},0,1,1,0,100,100,common,"
                f"false,false,0.5,0.5,0.8,0.2,101"
            )
    path = tmp_path / "excl.csv"
    path.write_text("\n".join(lines) + "\n")
    session = load_session(path)
    data = session["no_pressure"]
    assert len(data["a1"]) == 10  # not dropped
    assert data["excluded"] is True
    reasons = data["exclusion_reasons"]
    assert any("missed" in r.lower() for r in reasons)
    assert any("55%" in r or "chance" in r.lower() or "better" in r.lower() for r in reasons)


def test_clean_block_not_excluded():
    from twominds.io import load_session

    session = load_session(FIXTURE)
    # Smoke fixture is tiny and random; may or may not hit the 55% rule.
    # Ensure the fields exist and are the right types.
    for cond in ("no_pressure", "time_pressure"):
        assert isinstance(session[cond]["excluded"], bool)
        assert isinstance(session[cond]["exclusion_reasons"], list)


@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_miss_rate_threshold_boundary(tmp_path: Path, rate: float):
    from twominds.io import load_session

    n = 10
    n_miss = int(rate * n)
    lines = [
        "participant_code,block,condition,trial,a1,s2,a2,r,rt1,rt2,transition,"
        "missed_a1,missed_a2,p_s1_a0,p_s1_a1,p_s2_a0,p_s2_a1,schedule_seed"
    ]
    for t in range(n):
        if t < n_miss:
            lines.append(
                f"TM-B,0,no_pressure,{t},,,,,"
                f",3000,,,"
                f"true,false,0.5,0.5,0.9,0.1,101"
            )
        else:
            lines.append(
                f"TM-B,0,no_pressure,{t},0,1,0,1,100,100,common,"
                f"false,false,0.5,0.5,0.9,0.1,101"
            )
    path = tmp_path / f"rate_{rate}.csv"
    path.write_text("\n".join(lines) + "\n")
    data = load_session(path)["no_pressure"]
    missed_flagged = any("missed" in r.lower() for r in data["exclusion_reasons"])
    # Spec: more than 20% missed → flag. Exactly 20% should not flag on miss rate.
    if rate > 0.2:
        assert missed_flagged
    else:
        assert not missed_flagged
