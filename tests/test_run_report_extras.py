from gemma_swe.analysis.run_report import summarise, to_markdown


def _row(i, resolved, code, error=None, tokens=0):
    return {
        "instance_id": f"t{i}",
        "repo": "o/r",
        "resolved": resolved,
        "test_exit_code": code,
        "error": error,
        "total_tokens": tokens,
        "patch_chars": 10,
    }


def test_exit_codes_timeouts_and_tokens():
    rows = [
        _row(1, True, 0, tokens=100),
        _row(2, False, 2, tokens=300),
        _row(3, False, None, error="Agent exceeded session timeout (4.5 min)"),
    ]
    s = summarise(rows)
    assert s["test_exit_codes"] == {"0": 1, "2": 1, "None": 1}
    assert s["timeouts"] == 1
    assert s["median_total_tokens"] == 200
    assert "tests could not be collected" in to_markdown("x", s)


def test_rate_on_solvable_tasks_only():
    rows = [_row(1, True, 0), _row(2, False, 2), _row(3, False, 1)]
    s = summarise(rows, solvable={"t1", "t3"})
    assert s["on_solvable_tasks"]["n_tasks"] == 2
    assert s["on_solvable_tasks"]["resolved"] == 1
    assert "reference patch" in to_markdown("x", s)
    assert summarise(rows)["on_solvable_tasks"] is None
