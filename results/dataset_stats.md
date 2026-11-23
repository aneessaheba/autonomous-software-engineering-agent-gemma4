# Public task set: descriptive statistics

Computed over the 129 public development tasks in `tasks.jsonl`.
Regenerate with `gemma-swe stats`.

## Composition

| Repository | Tasks |
| :--- | ---: |
| fastapi/fastapi | 67 |
| Textualize/rich | 48 |
| psf/requests | 13 |
| encode/httpx | 1 |

Tasks with maintainer hints: 0 of 129.

## Size of the reference fix (non-test files only)

| Quantity | Median | Mean | P75 | P90 | Max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Source files touched | 1 | 2.2 | 2 | 4 | 26 |
| Lines added | 8 | 138.6 | 31 | 102 | 12170 |
| Lines removed | 2 | 29.2 | 8 | 24 | 1855 |
| Issue length (chars) | 418 | 803.6 | 1000 | 1891 | 10095 |

| Changed lines (added + removed) | Tasks |
| :--- | ---: |
| 1-5 | 43 |
| 6-20 | 38 |
| 21-100 | 32 |
| >100 | 16 |

## What the issue text gives away

| Property | Share of tasks |
| :--- | ---: |
| Fix is confined to one source file | 70.5% |
| Fix creates at least one new file | 8.5% |
| Fix touches a non-Python file | 0.0% |
| Issue names a gold file (path or basename) | 7.8% |
| Issue names a gold module stem as a word | 15.5% |

## Where fixes land

| Top-level directory | Gold files |
| :--- | ---: |
| `fastapi/fastapi` | 129 |
| `rich/rich` | 91 |
| `fastapi/docs_src` | 34 |
| `requests/src` | 15 |
| `httpx/src` | 7 |
| `fastapi/scripts` | 2 |
