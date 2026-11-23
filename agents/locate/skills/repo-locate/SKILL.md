---
name: repo-locate
description: Ranks the source files and functions most likely to need editing for an issue. Run it first, before any grep or file reading.
---

# repo-locate

Finds where to edit. It indexes every Python source file under `/workspace`,
splits it into functions, methods and classes, and ranks them against your
query. It takes under a second and never modifies the repository.

## How to run

Call `run_skill_script` with:

- `skill_name`: `repo-locate`
- `file_path`: `scripts/locate.py`
- `args`: a list holding one string, the query

The query should contain the identifiers, error messages and key phrases from
the issue: class, function and parameter names, option names, exception text.
More specific words give better results. Copy them from the issue; do not guess.

## Reading the output

The output lists up to eight files, best first, each with the functions that
matched and their line ranges. Open the first file with `read_file` using the
printed `start_line` and `end_line`. If the first file is clearly wrong, try
the next one, or run the script again with different identifiers.

Tests are excluded by default. Add `"--include-tests"` to `args` to rank test
files as well, which is useful for finding the test that covers a function.
