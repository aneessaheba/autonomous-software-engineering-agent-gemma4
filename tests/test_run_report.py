from gemma_swe.analysis.run_report import compare, summarise, wilson_interval


def _row(i, resolved, repo="o/r", calls=10, secs=100.0, patch=50):
    return {
        "instance_id": f"t{i}",
        "repo": repo,
        "resolved": resolved,
        "tool_calls": calls,
        "duration_seconds": secs,
        "patch_chars": patch,
    }


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(30, 129)
    assert lo < 30 / 129 < hi
    assert 0.16 < lo < 0.18 and 0.31 < hi < 0.32
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(0, 10)[0] == 0.0


def test_summarise_counts_and_groups():
    rows = [_row(1, True), _row(2, False, patch=0), _row(3, True, repo="o/other")]
    s = summarise(rows)
    assert (s["n_tasks"], s["resolved"], s["empty_patches"]) == (3, 2, 1)
    assert s["resolution_rate"] == round(2 / 3, 4)
    assert s["by_repo"] == {"o/other": {"n": 1, "resolved": 1}, "o/r": {"n": 2, "resolved": 1}}


def test_compare_is_paired_on_shared_tasks():
    base = [_row(1, True), _row(2, False), _row(3, False)]
    other = [_row(1, False), _row(2, True), _row(3, True), _row(4, True)]
    c = compare(base, other)
    assert c["n_shared"] == 3
    assert (c["gained"], c["lost"], c["net"]) == (["t2", "t3"], ["t1"], 1)
